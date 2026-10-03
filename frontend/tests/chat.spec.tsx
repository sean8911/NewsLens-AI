/** @jsxImportSource react */
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { test, expect } from "@playwright/test";
import { AnswerText, ChatTurn } from "../src/components/chat";
import { LocaleContext } from "../src/components/locale";
import {
  parseRecent,
  rememberRecent,
  mergeTurns,
  recoveredAttempt,
} from "../src/lib/conversations";
import type { ConversationHistory, ConversationTurn } from "../src/lib/types";

const id = "12345678-1234-1234-1234-123456789abc";
const turn: ConversationTurn = {
  sequence: 1,
  question: "What happened?",
  resolved_question: "What happened at OpenAI?",
  created_at: "2026-09-23T12:00:00Z",
  response: {
    answer: "The report says so [1]. Unknown [9]. <script>alert(1)</script>",
    status: "answered",
    reason: null,
    follow_up: true,
    sources: [
      {
        citation_id: 1,
        article_id: 29,
        title: "A report",
        source: "Publisher",
        published_at: "2026-09-22T12:00:00Z",
        url: "https://example.com/report",
        excerpts: ["Original supporting text"],
        translations: ["Translated text"],
      },
    ],
  },
};
const history: ConversationHistory = {
  conversation: {
    id,
    revision: 1,
    created_at: turn.created_at,
    updated_at: turn.created_at,
  },
  turns: [turn],
  next_before: null,
};

test("citations link only to supplied evidence and answer markup is escaped", () => {
  const html = renderToStaticMarkup(
    <AnswerText
      answer={turn.response.answer}
      sources={turn.response.sources}
      scope="first"
    />,
  );
  expect(html).toContain('href="#first-source-1"');
  expect(html).not.toContain('href="#first-source-9"');
  expect(html).toContain("Unknown [9]");
  expect(html).not.toContain("<script>");
});

test("Chinese chat labels preserve original answers and citation targets", () => {
  const html = renderToStaticMarkup(
    <LocaleContext.Provider value={{ locale: "zh", setLocale: () => {} }}>
      <ChatTurn turn={turn} conversationId={id} />
    </LocaleContext.Provider>,
  );
  for (const text of [
    "新闻来源",
    "复制回答",
    "查看原文依据",
    "本次追问是如何理解的",
    "已保存",
    "在新标签页中打开",
    "The report says so",
    "Original supporting text",
    "A report",
  ])
    expect(html).toContain(text);
  expect(html).toContain(`href="#chat-${id}-turn-1-source-1"`);
  expect(html).not.toContain("Grounded in reporting");
});
test("repeated citations in separate turns have distinct targets, dates and excerpts", () => {
  const html = renderToStaticMarkup(
    <>
      <ChatTurn turn={turn} conversationId={id} />
      <ChatTurn turn={{ ...turn, sequence: 2 }} conversationId={id} />
    </>,
  );
  for (const sequence of [1, 2]) {
    expect(html).toContain(`href="#chat-${id}-turn-${sequence}-source-1"`);
    expect(html).toContain(`id="chat-${id}-turn-${sequence}-source-1"`);
  }
  for (const text of [
    "Publisher",
    "A report",
    "2026-09-22",
    "Original supporting text",
    "Translated text",
    "/articles/29",
    "How this follow-up was understood",
    "What happened at OpenAI?",
  ])
    expect(html).toContain(text);
});
test("abstentions and clarification do not claim a grounded answer", () => {
  for (const [status, label] of [
    ["insufficient_information", "Not enough evidence"],
    ["needs_clarification", "Clarification needed"],
  ] as const) {
    const html = renderToStaticMarkup(
      <ChatTurn
        turn={{ ...turn, response: { ...turn.response, status, sources: [] } }}
        conversationId={id}
      />,
    );
    expect(html).toContain(label);
    expect(html).not.toContain("Grounded in reporting");
    expect(html).not.toContain('class="citation-card"');
  }
});
test("malformed and oversized recent-chat storage cannot break loading", () => {
  for (const raw of [
    null,
    "{broken",
    "null",
    "{}",
    "x".repeat(100001),
    JSON.stringify([{ id: "invalid", title: "bad", updatedAt: "today" }]),
  ])
    expect(parseRecent(raw)).toEqual([]);
  const recent = { id, title: "A".repeat(150), updatedAt: turn.created_at };
  expect(parseRecent(JSON.stringify([recent, recent, null]))).toEqual([
    { ...recent, title: "A".repeat(100) },
  ]);
});
test("recent chats retain first title, move to front and stay bounded", () => {
  const items = Array.from({ length: 20 }, (_, i) => ({
    id: `12345678-1234-1234-1234-${String(i).padStart(12, "0")}`,
    title: `Chat ${i}`,
    updatedAt: turn.created_at,
  }));
  const updated = rememberRecent(items, {
    ...items[5],
    title: "follow-up question",
  });
  expect(updated[0].title).toBe("Chat 5");
  expect(updated).toHaveLength(20);
  expect(
    rememberRecent(updated, { id, title: "new", updatedAt: turn.created_at }),
  ).toHaveLength(20);
});
test("overlapping history pages merge chronologically without duplicate turns", () => {
  const revised = { ...turn, question: "replacement" };
  expect(
    mergeTurns(
      [{ ...turn, sequence: 2 }, turn],
      [revised, { ...turn, sequence: 3 }],
    ),
  ).toEqual([revised, { ...turn, sequence: 2 }, { ...turn, sequence: 3 }]);
});
test("recovery only clears the exact attempted turn in its own conversation", () => {
  const attempt = { id, revision: 0, question: turn.question };
  expect(recoveredAttempt(history, attempt)).toBe(true);
  expect(recoveredAttempt(history, null)).toBe(false);
  expect(recoveredAttempt(history, { ...attempt, id: "another" })).toBe(false);
  expect(recoveredAttempt(history, { ...attempt, revision: 1 })).toBe(false);
  expect(recoveredAttempt(history, { ...attempt, question: "different" })).toBe(
    false,
  );
});
