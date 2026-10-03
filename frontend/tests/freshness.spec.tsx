/** @jsxImportSource react */
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { test, expect } from "@playwright/test";
import { FreshnessPanel, AnswerFreshness } from "../src/components/freshness";
import { relativeTime, updateLabel } from "../src/lib/freshness";
import { LocaleContext } from "../src/components/locale";
import type { Status, ChatSource } from "../src/lib/types";

const now = Date.parse("2026-09-23T12:00:00Z");
test("Chinese freshness keeps failed attempts and stale coverage explicit", () => {
  const html = renderToStaticMarkup(
    <LocaleContext.Provider value={{ locale: "zh", setLocale: () => {} }}>
      <FreshnessPanel
        data={status}
        error={false}
        loading={false}
        now={now}
        refresh={() => {}}
      />
      <AnswerFreshness sources={[source]} />
    </LocaleContext.Provider>,
  );
  for (const text of [
    "5 分钟前",
    "部分更新",
    "最近一次更新尝试：",
    "失败",
    "不会实时浏览互联网",
    "不代表已覆盖更新的新闻",
  ])
    expect(html).toContain(text);
  expect(relativeTime(null, now, "zh")).toBe("暂无记录");
  expect(updateLabel("2026-09-23T10:00:00Z", now, "zh")).toBe("新闻待更新");
});
const status: Status = {
  status: "ok",
  database: "available",
  article_count: 10,
  checked_at: "2026-09-23T12:00:00Z",
  last_successful_news_update: "2026-09-23T11:55:00Z",
  last_successful_update_status: "partial",
  latest_article_published_at: "2026-09-08T04:57:00Z",
  latest_article_retrieved_at: "2026-09-23T11:54:00Z",
  last_update_attempt_at: "2026-09-23T11:59:00Z",
  last_update_attempt_status: "failed",
};
const source: ChatSource = {
  citation_id: 1,
  article_id: 1,
  title: "Report",
  source: "Publisher",
  url: "https://example.com",
  published_at: "2026-09-08T04:57:00Z",
  retrieved_at: "2026-09-09T05:02:00Z",
  excerpts: ["Evidence"],
};
function panel(
  data: Status | undefined = status,
  error = false,
  loading = false,
) {
  return renderToStaticMarkup(
    <FreshnessPanel
      data={data}
      error={error}
      loading={loading}
      now={now}
      refresh={() => {}}
    />,
  );
}

test("freshness separates update time, old publication and partial or failed attempts", () => {
  const html = panel();
  for (const value of [
    "5 minutes ago",
    "23 Sept 2026, 11:55 UTC",
    "8 Sept 2026, 04:57 UTC",
    "Partial update",
    "Latest update attempt: failed",
    "not the live web",
    "30 minutes",
  ])
    expect(html).toContain(value);
  expect(html).not.toContain("Live news");
});
test("unavailable and failed status checks never present cached dates as current", () => {
  const failed = panel(status, true);
  expect(failed).toContain("Freshness unavailable");
  expect(failed).toContain("last known values");
  const degraded = panel({
    ...status,
    status: "degraded",
    database: "unavailable",
  });
  expect(degraded).toContain("coverage cannot be confirmed");
  expect(degraded).not.toContain("5 minutes ago");
});
test("empty, unknown and future dates have explicit states", () => {
  const empty = panel({
    ...status,
    article_count: 0,
    last_successful_news_update: null,
    latest_article_published_at: null,
  });
  expect(empty).toContain("No successful update recorded");
  expect(empty).toContain("No articles collected");
  expect(panel({ ...status, latest_article_published_at: null })).toContain(
    "Publication date unknown",
  );
  expect(
    panel({ ...status, latest_article_published_at: "2099-01-01T00:00:00Z" }),
  ).toContain("Future publication timestamp");
});
test("relative times handle boundaries, long gaps and clock errors", () => {
  expect(relativeTime("2026-09-23T11:59:30Z", now)).toBe(
    "Less than a minute ago",
  );
  expect(relativeTime("2026-09-23T11:59:00Z", now)).toBe("1 minute ago");
  expect(relativeTime("2026-09-23T10:00:00Z", now)).toBe("2 hours ago");
  expect(relativeTime("2026-09-08T12:00:00Z", now)).toBe("15 days ago");
  expect(relativeTime("invalid", now)).toBe("Not recorded");
  expect(relativeTime("2026-09-24T00:00:00Z", now)).toContain(
    "Future timestamp",
  );
  expect(updateLabel("2026-09-23T11:30:00Z", now)).toBe("Update overdue");
  expect(updateLabel("2026-09-23T11:30:01Z", now)).toBe(
    "Collection recently checked",
  );
  expect(updateLabel(null, now)).toBe("No successful update recorded");
});
test("answer cutoff is scoped to its saved sources, with legacy and abstention fallbacks", () => {
  const html = renderToStaticMarkup(
    <AnswerFreshness
      sources={[
        source,
        { ...source, citation_id: 2, retrieved_at: "2026-09-10T05:02:00Z" },
      ]}
    />,
  );
  expect(html).toContain("10 Sept 2026, 05:02 UTC");
  expect(html).toContain("8 Sept 2026, 04:57 UTC");
  expect(html).toContain("not a claim to cover newer news");
  expect(
    renderToStaticMarkup(
      <AnswerFreshness
        sources={[source, { ...source, retrieved_at: undefined }]}
      />,
    ),
  ).toContain("not recorded for every source");
  expect(renderToStaticMarkup(<AnswerFreshness sources={[]} />)).toContain(
    "collection cutoff is unavailable",
  );
});
