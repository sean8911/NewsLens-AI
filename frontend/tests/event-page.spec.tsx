/** @jsxImportSource react */
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { test, expect } from "@playwright/test";
import {
  SummarySection,
  ComparisonContent,
  TimelineContent,
  EventSidebar,
} from "../src/components/event-intelligence";
import { event, comparison, timeline } from "./fixtures/event";

test("summary evidence and comparison evidence have separate anchors even when citation IDs overlap", () => {
  const html = renderToStaticMarkup(
    <>
      <SummarySection event={event} />
      <ComparisonContent value={comparison} />
    </>,
  );
  expect(html).toContain('href="#summary-citation-1"');
  expect(html).toContain('id="summary-citation-1"');
  expect(html).toContain('href="#comparison-citation-1"');
  expect(html).toContain('id="comparison-citation-1"');
  expect(html).toContain("Comparison Publisher");
  expect(html).toContain("Test Publisher");
  expect(html).not.toContain("Legacy summary");
  expect(html).toContain("outside this summary");
});

test("stale summary, comparisons, timelines and entities never present old insight as current", () => {
  for (const staleEvent of [
    { ...event, stale: true },
    { ...event, event_summary: { ...event.event_summary, stale: true } },
  ]) {
    const html = renderToStaticMarkup(<SummarySection event={staleEvent} />);
    expect(html).not.toContain("Ten units were reported");
    expect(html).toContain("out of date");
  }
  for (const view of [
    { ...comparison, stale: true },
    { ...comparison, status: "stale" as const },
  ]) {
    const html = renderToStaticMarkup(<ComparisonContent value={view} />);
    expect(html).not.toContain("Comparison Publisher");
  }
  expect(
    renderToStaticMarkup(<ComparisonContent value={comparison} eventStale />),
  ).not.toContain("Ten units were reported");
  expect(
    renderToStaticMarkup(
      <TimelineContent value={timeline} eventStale onPage={() => {}} />,
    ),
  ).not.toContain("Dated evidence");
  const timed = renderToStaticMarkup(
    <TimelineContent
      value={{ ...timeline, status: "stale" }}
      onPage={() => {}}
    />,
  );
  expect(timed).not.toContain("Dated evidence");
  const sidebar = renderToStaticMarkup(
    <EventSidebar event={{ ...event, stale: true }} />,
  );
  expect(sidebar).not.toContain("Fixture organization");
  expect(sidebar).toContain("Fixture report");
});

test("missing comparisons do not invent agreement or contradictions", () => {
  const html = renderToStaticMarkup(<ComparisonContent value={comparison} />);
  expect(html).toContain("This does not mean the sources disagree");
  expect(html).toContain("not proven contradictions");
  expect(html).toContain("Unmatched wording");
  const unavailable = renderToStaticMarkup(
    <ComparisonContent
      value={{ ...comparison, status: "insufficient_sources", document: null }}
    />,
  );
  expect(unavailable).toContain("At least two independent sources");
  expect(unavailable).not.toContain("Ten units were reported");
});

test("numeric differences and emphasis retain their cited reports and limits", () => {
  const value = structuredClone(comparison);
  value.document!.differences = [
    {
      kind: "numbers",
      explanation: "Reported totals differ.",
      reports: [{ text: "Eleven units reported.", citation_ids: [1] }],
    },
  ];
  value.document!.emphasis = [
    {
      topic: "cost",
      explanation: "Cost terms appear in this excerpt.",
      reports: [{ text: "A cost detail.", citation_ids: [1] }],
      other_compared_sources: ["Other Publisher"],
    },
  ];
  const html = renderToStaticMarkup(<ComparisonContent value={value} />);
  expect(html).toContain("Eleven units reported");
  expect(html).toContain("A cost detail");
  expect(html).toContain("Other Publisher");
  expect(html).toContain("does not establish an omission");
});

test("timeline separates undated evidence and exposes pagination and exclusions", () => {
  const html = renderToStaticMarkup(
    <TimelineContent value={timeline} onPage={() => {}} />,
  );
  expect(html).toContain("Article publication times in UTC");
  expect(html).toContain("Undated reports");
  expect(html).toContain("Publication date unavailable");
  expect(html).not.toContain("1970");
  expect(html).toContain("1–2");
  expect(html).toContain("of 12 reports");
  expect(html).toContain("not enough usable text");
  expect(html).toMatch(/disabled=""[^>]*>Earlier reports/);
  expect(html).not.toMatch(/disabled=""[^>]*>Later reports/);
});

test("entity links, source links and question prefills preserve provenance and escape content", () => {
  const value = {
    ...event,
    title: "<script>alert(1)</script>",
    related_entities: [
      { ...event.related_entities[0], name: "<img src=x onerror=alert(1)>" },
    ],
  };
  const html = renderToStaticMarkup(<EventSidebar event={value} />);
  expect(html).toContain("/feed?entity_id=7");
  expect(html).toContain('href="https://example.com/report"');
  expect(html).toContain('rel="noopener noreferrer"');
  expect(html).toContain("%3Cscript%3E");
  expect(html).not.toContain("<img");
  expect(html).not.toContain("<script");
});
