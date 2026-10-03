import { test, expect } from "@playwright/test";
import { allowedOrigin } from "../src/lib/origin";

test("configured HTTPS origin works behind an internal HTTP proxy", () => {
  expect(
    allowedOrigin(
      "https://news.example.com",
      "127.0.0.1:3000",
      "http:",
      "https://news.example.com",
    ),
  ).toBe(true);
  expect(
    allowedOrigin(
      "https://evil.example",
      "evil.example",
      "https:",
      "https://news.example.com",
    ),
  ).toBe(false);
});

test("invalid configured origins fail closed and local origin checks remain strict", () => {
  for (const configured of [
    "",
    "http://news.example.com",
    "https://news.example.com/",
    "https://user:password@news.example.com",
    "https://news.example.com/path",
  ]) {
    expect(
      allowedOrigin(
        "https://news.example.com",
        "news.example.com",
        "https:",
        configured,
      ),
    ).toBe(false);
  }
  expect(
    allowedOrigin("http://localhost:3000", "localhost:3000", "http:"),
  ).toBe(true);
  expect(allowedOrigin("http://evil.example", "localhost:3000", "http:")).toBe(
    false,
  );
  expect(allowedOrigin("null", "localhost:3000", "http:")).toBe(false);
});
