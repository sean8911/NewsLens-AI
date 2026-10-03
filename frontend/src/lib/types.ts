export type Entity = { id: number; name: string; entity_type: string };
export type Article = {
  id: number;
  title: string;
  description: string | null;
  summary: string | null;
  source: string;
  url: string;
  category: string;
  language: string | null;
  published_at: string | null;
  retrieved_at: string;
  author: string | null;
  article_topics: { topic: string; confidence: number }[];
  entities: Entity[];
  content?: string | null;
  content_available?: boolean;
};
export type Page<T> = {
  results: T[];
  total: number;
  count: number;
  limit: number;
  offset: number;
  has_more: boolean;
};
export type Topic = { topic: string; article_count: number };
export type Status = {
  status: "ok" | "degraded";
  database: string;
  checked_at: string;
  latest_article_published_at: string | null;
  latest_article_retrieved_at: string | null;
  last_successful_update_status: "succeeded" | "partial" | null;
  last_update_attempt_at: string | null;
  last_update_attempt_status:
    "running" | "succeeded" | "partial" | "failed" | "interrupted" | null;
  article_count: number | null;
  last_successful_news_update: string | null;
};
export type NewsEvent = {
  id: number;
  title: string;
  summary: string | null;
  source_count: number;
  article_count: number;
  last_updated_at: string;
  last_published_at: string | null;
  stale: boolean;
  related_entities: (Entity & { article_ids: number[] })[];
  event_summary: {
    status: string;
    generated_at: string | null;
    stale: boolean;
    document: SummaryDocument | null;
  };
  articles: {
    article_id: number;
    title: string;
    source: string;
    url: string;
    published_at: string | null;
    stale: boolean;
  }[];
};
export type EventPage = { results: NewsEvent[]; total: number; count: number };
export type Trend = {
  id: string;
  label: string;
  topic: string;
  score: number;
  article_count: number;
  source_count: number;
  growth_percent: number | null;
  state: string;
  sources: string[];
  entities: { name: string }[];
  evidence: {
    article_id: number;
    title: string;
    source: string;
    url: string;
    published_at: string;
  }[];
};
export type Trends = {
  status: string;
  as_of: string;
  window_hours: number;
  current_articles: number;
  results: Trend[];
  count: number;
};
export type SearchHit = {
  article_id: number;
  title: string;
  source: string;
  published_at: string | null;
  summary: string | null;
  url: string;
  category: string;
  relevance_score: number;
};
export type SearchResult = {
  results: SearchHit[];
  count: number;
  method: string;
};
export type ChatSource = {
  retrieved_at?: string | null;
  citation_id: number;
  article_id: number;
  title: string;
  source: string;
  url: string;
  published_at: string | null;
  excerpts: string[];
  translations?: (string | null)[];
};
export type ChatAnswer = {
  answer: string;
  conversation_id?: string | null;
  conversation_revision?: number | null;
  resolved_question?: string | null;
  follow_up?: boolean;
  answer_language?: string;
  status: "answered" | "insufficient_information" | "needs_clarification";
  sources: ChatSource[];
  reason: string | null;
};

export type EvidenceCitation = {
  citation_id: number;
  article_id: number;
  source: string;
  title: string;
  url: string;
  published_at: string | null;
  excerpt: string;
};
export type CitedFact = { text: string; citation_ids: number[] };
export type SummaryDocument = {
  what_happened: CitedFact;
  key_facts: CitedFact[];
  latest_update: CitedFact | null;
  latest_reported_at: string | null;
  organizations: { name: string; citation_ids: number[] }[];
  people: { name: string; citation_ids: number[] }[];
  citations: EvidenceCitation[];
  source_count: number;
  article_count: number;
  total_event_articles: number;
  omitted_article_ids: number[];
};
export type ComparisonDocument = {
  agreement: CitedFact[];
  differences: {
    kind: "numbers" | "wording";
    explanation: string;
    reports: CitedFact[];
  }[];
  additional_details: CitedFact[];
  emphasis: {
    topic: string;
    explanation: string;
    reports: CitedFact[];
    other_compared_sources: string[];
  }[];
  citations: EvidenceCitation[];
  source_count: number;
  article_count: number;
  total_event_articles: number;
  omitted_article_ids: number[];
  scope: string[];
};
export type EventComparison = {
  event_id: number;
  status:
    | "compared"
    | "insufficient_sources"
    | "insufficient_text"
    | "unsupported_language"
    | "stale";
  stale: boolean;
  document: ComparisonDocument | null;
  computed_at: string;
  notes: string[];
};
export type TimelineItem = {
  id: string;
  timestamp: string | null;
  timestamp_basis: "publication" | "unknown";
  description: string;
  source_article_id: number;
  citation: EvidenceCitation;
};
export type TimelineDocument = {
  items: TimelineItem[];
  undated_items: TimelineItem[];
  excluded_articles: { article_id: number; reason: string }[];
  count: number;
  total_count: number;
  dated_count: number;
  undated_count: number;
  total_event_articles: number;
  source_count: number;
  limit: number;
  offset: number;
  next_offset: number | null;
  scope: string[];
};
export type EventTimeline = {
  event_id: number;
  status: "ready" | "insufficient_evidence" | "stale";
  stale: boolean;
  document: TimelineDocument | null;
  computed_at: string;
  notes: string[];
};

export type ConversationInfo = {
  id: string;
  revision: number;
  created_at: string;
  updated_at: string;
};
export type ConversationTurn = {
  sequence: number;
  question: string;
  resolved_question: string;
  response: ChatAnswer;
  created_at: string;
};
export type ConversationHistory = {
  conversation: ConversationInfo;
  turns: ConversationTurn[];
  next_before: number | null;
};
