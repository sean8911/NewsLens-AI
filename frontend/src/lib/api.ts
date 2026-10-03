"use client";

import { useCallback, useEffect, useState } from "react";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    cache: "no-store",
    ...options,
  });
  if (!response.ok) {
    const message =
      response.status === 404
        ? "This story could not be found."
        : response.status === 429
          ? "Too many requests. Wait a moment before trying again."
          : response.status === 503 && response.headers.has("retry-after")
            ? "The news service is busy, possibly updating its index. Wait a moment, then reload saved history before sending again."
            : response.status === 422
              ? "Check your filters or question and try again."
              : response.status === 504
                ? "This request took too long. Please try again."
                : "The news service is unavailable. Please try again in a moment.";
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export function useApi<T>(path: string) {
  const [result, setResult] = useState<{
    path: string;
    data?: T;
    error?: string;
  }>({ path: "" });
  const [revision, setRevision] = useState(0);
  const retry = useCallback(() => {
    setResult({ path: "" });
    setRevision((value) => value + 1);
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    api<T>(path, { signal: controller.signal }).then(
      (data) => {
        if (!controller.signal.aborted) setResult({ path, data });
      },
      (error) => {
        if (!controller.signal.aborted)
          setResult({ path, error: error.message });
      },
    );
    return () => controller.abort();
  }, [path, revision]);
  return {
    data: result.path === path ? result.data : undefined,
    error: result.path === path ? result.error : undefined,
    loading: result.path !== path,
    retry,
  };
}

export function query(values: Record<string, string | number | undefined>) {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== "") params.set(key, String(value));
  });
  return params.toString();
}

export function date(value: string | null, withTime = false, locale = "en") {
  const unavailable = locale === "zh" ? "日期未知" : "Date unavailable";
  if (!value) return unavailable;
  const parsed = new Date(value);
  if (!Number.isFinite(parsed.getTime())) return unavailable;
  return (
    new Intl.DateTimeFormat(locale === "zh" ? "zh-CN" : "en-GB", {
      day: "numeric",
      month: "short",
      year: "numeric",
      timeZone: "UTC",
      ...(withTime ? ({ hour: "2-digit", minute: "2-digit" } as const) : {}),
    }).format(parsed) + (withTime ? " UTC" : "")
  );
}

export function safeUrl(value: string) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : undefined;
  } catch {
    return undefined;
  }
}
