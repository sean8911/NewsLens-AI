import { createServer, type Server } from "node:http";
import { test, expect } from "@playwright/test";

let backend: Server;
const calls: { url: string; body: unknown; cookie?: string }[] = [];
test.beforeAll(async () => {
  backend = createServer(async (req, res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = chunks.length
      ? JSON.parse(Buffer.concat(chunks).toString())
      : null;
    calls.push({ url: req.url || "", body, cookie: req.headers.cookie });
    res.setHeader("Content-Type", "application/json");
    if (req.method === "POST" && req.url === "/conversations") {
      res.statusCode = 201;
      res.end(
        JSON.stringify({
          id: "12345678-1234-1234-1234-123456789abc",
          revision: 0,
        }),
      );
    } else if (body?.q === "conflict") {
      res.statusCode = 409;
      res.end(JSON.stringify({ detail: "private revision detail" }));
    } else if (["limited", "busy", "bad-retry"].includes(body?.q)) {
      res.statusCode = body.q === "busy" ? 503 : 429;
      res.setHeader(
        "Retry-After",
        body.q === "bad-retry" ? "private-detail" : "12",
      );
      res.end(JSON.stringify({ detail: "private quota information" }));
    } else if (body?.q === "fail") {
      res.statusCode = 500;
      res.end(JSON.stringify({ detail: "secret database credentials" }));
    } else if (body?.q === "invalid") {
      res.statusCode = 422;
      res.end(JSON.stringify({ detail: "private internal information" }));
    } else res.end(JSON.stringify({ ok: true, url: req.url, body }));
  });
  await new Promise<void>((resolve) =>
    backend.listen(
      Number(process.env.NEWSLENS_TEST_BACKEND_PORT || 8010),
      "127.0.0.1",
      resolve,
    ),
  );
});
test.afterAll(async () => {
  await new Promise<void>((resolve, reject) =>
    backend.close((error) => (error ? reject(error) : resolve())),
  );
});
test.beforeEach(() => {
  calls.length = 0;
});

test("rate and capacity limits preserve safe retry guidance", async ({
  request,
}) => {
  for (const q of ["limited", "busy", "bad-retry"]) {
    const response = await request.post("/api/search", { data: { q } });
    expect(response.status()).toBe(q === "busy" ? 503 : 429);
    expect(response.headers()["retry-after"]).toBe(
      q === "bad-retry" ? undefined : "12",
    );
    expect(response.headers()["cache-control"]).toBe("no-store");
    expect(await response.text()).not.toContain("private");
  }
});

test("article query parameters are preserved and private headers are not forwarded", async ({
  request,
}) => {
  const result = await request.get(
    "/api/articles?source=BBC%20News&limit=12&offset=12",
    { headers: { Cookie: "private=session" } },
  );
  expect(result.status()).toBe(200);
  expect(calls).toHaveLength(1);
  const forwarded = new URL(calls[0].url, "http://backend.invalid");
  expect(forwarded.pathname).toBe("/articles");
  expect(Object.fromEntries(forwarded.searchParams)).toEqual({
    source: "BBC News",
    limit: "12",
    offset: "12",
  });
  expect(calls[0].cookie).toBeUndefined();
  expect(result.headers()["cache-control"]).toBe("no-store");
});
test("JSON search and chat bodies reach only their declared endpoint", async ({
  request,
}) => {
  for (const endpoint of ["search", "chat"]) {
    const result = await request.post(`/api/${endpoint}`, {
      data: { q: "news" },
    });
    expect(result.status()).toBe(200);
    expect((await result.json()).body).toEqual({ q: "news" });
  }
  expect(calls.map((call) => call.url)).toEqual(["/search", "/chat"]);
});
test("same-origin browser POST is accepted", async ({ request }) => {
  const result = await request.post("/api/search", {
    data: { q: "news" },
    headers: { Origin: "http://127.0.0.1:3001" },
  });
  expect(result.status()).toBe(200);
});
test("cross-origin browser POST is rejected before upstream", async ({
  request,
}) => {
  const result = await request.post("/api/chat", {
    data: { question: "news" },
    headers: { Origin: "https://unrelated.example" },
  });
  expect(result.status()).toBe(403);
  expect(calls).toHaveLength(0);
});
test("unlisted endpoints and methods are not forwarded", async ({
  request,
}) => {
  for (const endpoint of [
    "admin",
    "https://example.com",
    "articles/not-an-id",
    "search",
    "events/1/comparison/admin",
    "events/0/timeline",
  ]) {
    expect((await request.get(`/api/${endpoint}`)).status()).toBe(404);
  }
  expect((await request.post("/api/articles", { data: {} })).status()).toBe(
    404,
  );
  expect((await request.delete("/api/articles/1")).status()).toBe(405);
  expect(calls).toHaveLength(0);
});
test("invalid or oversized JSON is rejected before upstream", async ({
  request,
}) => {
  expect(
    (
      await request.post("/api/chat", {
        data: Buffer.from("{bad"),
        headers: { "Content-Type": "application/json" },
      })
    ).status(),
  ).toBe(422);
  expect((await request.post("/api/chat", { data: "text" })).status()).toBe(
    415,
  );
  expect(
    (
      await request.post("/api/chat", {
        data: { question: "x".repeat(17_000) },
      })
    ).status(),
  ).toBe(413);
  for (const body of [null, [], "plain string", 123]) {
    expect(
      (
        await request.post("/api/chat", {
          data: Buffer.from(JSON.stringify(body)),
          headers: { "Content-Type": "application/json" },
        })
      ).status(),
    ).toBe(422);
  }
  expect(calls).toHaveLength(0);
});
test("upstream errors are sanitized while validation status is preserved", async ({
  request,
}) => {
  const unavailable = await request.post("/api/search", {
    data: { q: "fail" },
  });
  expect(unavailable.status()).toBe(503);
  expect(await unavailable.text()).not.toContain("secret");
  const invalid = await request.post("/api/search", { data: { q: "invalid" } });
  expect(invalid.status()).toBe(422);
  expect(await invalid.text()).not.toContain("private");
});
test("all frontend pages render without depending on live models", async ({
  request,
}) => {
  for (const path of [
    "/",
    "/feed",
    "/category/AI",
    "/articles/1",
    "/events",
    "/events/1",
    "/search",
    "/chat",
    "/trending",
  ]) {
    const result = await request.get(path);
    expect(result.status()).toBe(200);
    expect(await result.text()).toContain("NewsLens");
  }
  expect((await request.get("/not-a-page")).status()).toBe(404);
});

test("event comparison and timeline reads preserve their scope and paging", async ({
  request,
}) => {
  for (const endpoint of [
    "events/29/comparison",
    "events/29/timeline?limit=10&offset=20",
  ]) {
    const response = await request.get(`/api/${endpoint}`);
    expect(response.status()).toBe(200);
    expect((await response.json()).url).toBe(`/${endpoint}`);
  }
  expect(
    (await request.post("/api/events/29/timeline", { data: {} })).status(),
  ).toBe(404);
  expect(calls).toHaveLength(2);
});

const conversationId = "12345678-1234-1234-1234-123456789abc";
test("conversation creation preserves 201 and sends no body", async ({
  request,
}) => {
  const response = await request.post("/api/conversations", {
    headers: { Origin: "http://127.0.0.1:3001" },
  });
  expect(response.status()).toBe(201);
  expect((await response.json()).id).toBe(conversationId);
  expect(calls).toEqual([
    { url: "/conversations", body: null, cookie: undefined },
  ]);
});
test("conversation history preserves paging and only permits exact UUID routes", async ({
  request,
}) => {
  const path = `/conversations/${conversationId}?limit=20&before=5`;
  expect((await request.get(`/api${path}`)).status()).toBe(200);
  expect(calls[0].url).toBe(path);
  for (const path of [
    "conversations",
    "conversations/not-a-uuid",
    `conversations/${conversationId}/delete`,
  ])
    expect((await request.get(`/api/${path}`)).status()).toBe(404);
  expect(calls).toHaveLength(1);
});
test("conversation creation rejects bodies and foreign origins", async ({
  request,
}) => {
  expect(
    (await request.post("/api/conversations", { data: {} })).status(),
  ).toBe(422);
  expect(
    (
      await request.post("/api/conversations", {
        headers: { Origin: "https://unrelated.example" },
      })
    ).status(),
  ).toBe(403);
  expect(calls).toHaveLength(0);
});
test("revision conflicts remain recoverable without exposing backend details", async ({
  request,
}) => {
  const response = await request.post("/api/chat", { data: { q: "conflict" } });
  expect(response.status()).toBe(409);
  expect(await response.text()).not.toContain("private");
});
