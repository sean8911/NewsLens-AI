import type { ConversationTurn, ConversationHistory } from "./types";
export const conversationIdPattern =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
export const recentKey = "newslens.recent-conversations.v1";
export type RecentConversation = {
  id: string;
  title: string;
  updatedAt: string;
};
export type Attempt = { id: string; revision: number; question: string };
export function parseRecent(raw: string | null): RecentConversation[] {
  if (!raw || raw.length > 100_000) return [];
  try {
    const value: unknown = JSON.parse(raw);
    if (!Array.isArray(value)) return [];
    const valid = value.filter(
      (item) =>
        item &&
        typeof item.id === "string" &&
        conversationIdPattern.test(item.id) &&
        typeof item.title === "string" &&
        typeof item.updatedAt === "string" &&
        Number.isFinite(Date.parse(item.updatedAt)),
    );
    return [
      ...new Map(
        valid.map((item) => [
          item.id,
          {
            id: item.id,
            title: item.title.slice(0, 100),
            updatedAt: item.updatedAt,
          },
        ]),
      ).values(),
    ].slice(0, 20);
  } catch {
    return [];
  }
}
export function rememberRecent(
  items: RecentConversation[],
  next: RecentConversation,
) {
  const existing = items.find((item) => item.id === next.id);
  return [
    { ...next, title: existing?.title || next.title.slice(0, 100) },
    ...items.filter((item) => item.id !== next.id),
  ].slice(0, 20);
}
export function mergeTurns(
  older: ConversationTurn[],
  newer: ConversationTurn[],
) {
  return [
    ...new Map(
      [...older, ...newer].map((turn) => [turn.sequence, turn]),
    ).values(),
  ].sort((a, b) => a.sequence - b.sequence);
}
export function recoveredAttempt(
  history: ConversationHistory,
  attempt: Attempt | null,
) {
  return (
    !!attempt &&
    history.conversation.id === attempt.id &&
    history.turns.some(
      (turn) =>
        turn.sequence === attempt.revision + 1 &&
        turn.question === attempt.question,
    )
  );
}
