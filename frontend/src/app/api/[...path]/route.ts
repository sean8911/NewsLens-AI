import { NextRequest, NextResponse } from "next/server";
import { allowedOrigin } from "@/lib/origin";

export const dynamic = "force-dynamic";
const allowedGet =
  /^(health|system\/status|articles(?:\/[1-9]\d{0,18})?|topics|entities(?:\/[1-9]\d{0,18})?|events(?:\/[1-9]\d{0,18}(?:\/(?:comparison|timeline))?)?|trending|conversations\/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$/;

async function proxy(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  const endpoint = path.join("/");
  const post = request.method === "POST";
  if (
    post
      ? !["search", "chat", "conversations"].includes(endpoint)
      : !allowedGet.test(endpoint)
  ) {
    return NextResponse.json({ detail: "Endpoint not found" }, { status: 404 });
  }
  const origin = request.headers.get("origin");
  if (post && origin) {
    const sameOrigin = allowedOrigin(
      origin,
      request.headers.get("host"),
      request.nextUrl.protocol,
      process.env.NEWSLENS_PUBLIC_ORIGIN,
    );
    if (!sameOrigin) {
      return NextResponse.json(
        { detail: "Origin not allowed" },
        { status: 403 },
      );
    }
  }
  try {
    const base = new URL(
      process.env.NEWSLENS_API_URL || "http://127.0.0.1:8000",
    );
    if (!["http:", "https:"].includes(base.protocol))
      throw new Error("Invalid backend configuration");
    let body: string | undefined;
    if (post && endpoint === "conversations" && request.body) {
      const reader = request.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value.byteLength) {
          await reader.cancel();
          return NextResponse.json(
            { detail: "No body expected" },
            { status: 422 },
          );
        }
      }
    }
    if (post && endpoint !== "conversations") {
      if (!request.headers.get("content-type")?.includes("application/json")) {
        return NextResponse.json({ detail: "JSON required" }, { status: 415 });
      }
      const reader = request.body?.getReader();
      const chunks: Uint8Array[] = [];
      let length = 0;
      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          length += value.byteLength;
          if (length > 16_384) {
            await reader.cancel();
            return NextResponse.json(
              { detail: "Request too large" },
              { status: 413 },
            );
          }
          chunks.push(value);
        }
      }
      body = Buffer.concat(chunks).toString("utf8");
      try {
        const parsed: unknown = JSON.parse(body);
        if (
          parsed === null ||
          typeof parsed !== "object" ||
          Array.isArray(parsed)
        ) {
          return NextResponse.json(
            { detail: "JSON object required" },
            { status: 422 },
          );
        }
      } catch {
        return NextResponse.json({ detail: "Invalid JSON" }, { status: 422 });
      }
    }
    const url = new URL(`/${endpoint}`, base);
    url.search = request.nextUrl.search;
    const response = await fetch(url, {
      method: request.method,
      body,
      cache: "no-store",
      redirect: "error",
      headers: body ? { "Content-Type": "application/json" } : {},
      signal: AbortSignal.timeout(post ? 180_000 : 15_000),
    });
    if (!response.ok) {
      const status = [404, 409, 413, 422, 429].includes(response.status)
        ? response.status
        : 503;
      const retry = response.headers.get("retry-after");
      const headers: Record<string, string> = { "Cache-Control": "no-store" };
      if (
        (status === 429 || status === 503) &&
        retry &&
        /^\d{1,4}$/.test(retry)
      ) {
        headers["Retry-After"] = String(
          Math.min(3600, Math.max(1, Number(retry))),
        );
      }
      return NextResponse.json(
        {
          detail:
            status === 429
              ? "Too many requests. Try again shortly."
              : "Request could not be completed",
        },
        { status, headers },
      );
    }
    return NextResponse.json(await response.json(), {
      status: response.status,
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    const timeout = error instanceof Error && error.name === "TimeoutError";
    return NextResponse.json(
      { detail: "News service unavailable" },
      { status: timeout ? 504 : 503 },
    );
  }
}

export { proxy as GET, proxy as POST };
