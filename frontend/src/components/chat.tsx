/** @jsxImportSource react */
"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useReducer, useRef, useState } from "react";
import {
  ArrowRight,
  BookOpen,
  Check,
  Copy,
  Sparkles,
  Plus,
  MessageSquare,
  RotateCcw,
  ChevronUp,
} from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type {
  ChatAnswer,
  ConversationInfo,
  ConversationHistory,
  ConversationTurn,
  ChatSource,
} from "@/lib/types";
import {
  conversationIdPattern,
  parseRecent,
  rememberRecent,
  mergeTurns,
  recoveredAttempt,
  recentKey,
  type RecentConversation,
  type Attempt,
} from "@/lib/conversations";
import { Loading, SourceLink } from "./ui";
import { AnswerFreshness } from "./freshness";
import { useLocale, type Locale } from "./locale";

const chineseMessages: Record<string, string> = {
  "Too many requests. Wait a moment before trying again.":
    "请求过于频繁，请稍后再试。",
  "The news service is busy, possibly updating its index. Wait a moment, then reload saved history before sending again.":
    "新闻服务正忙，可能正在更新索引。请稍等，重新加载历史记录后再发送。",
  "This conversation could not be found. Start a new conversation or choose another recent chat.":
    "找不到此对话。请新建对话或选择其他最近对话。",
  "This conversation changed elsewhere or reached its turn limit. Reload saved history before continuing.":
    "此对话已在其他窗口更改，或已达到轮数上限。请重新加载历史记录后继续。",
  "This conversation changed elsewhere. Reload saved history before continuing.":
    "此对话已在其他窗口更改。请重新加载历史记录后继续。",
  "The request could not be completed. Check the connection and reload saved history before sending again.":
    "无法完成请求。请检查连接，并重新加载历史记录后再发送。",
  "This conversation link is invalid. Start a new conversation.":
    "此对话链接无效，请新建对话。",
  "Your saved answer was recovered. You can continue the conversation.":
    "已恢复保存的回答，可以继续对话。",
  "Saved history refreshed. Review your draft before sending.":
    "历史记录已刷新。请检查草稿后再发送。",
  "Check your filters or question and try again.":
    "请检查筛选条件或问题后重试。",
  "This request took too long. Please try again.": "此次请求耗时过长，请重试。",
  "The news service is unavailable. Please try again in a moment.":
    "新闻服务暂时不可用，请稍后重试。",
};
function localizeMessage(message: string, locale: Locale) {
  return locale === "zh"
    ? chineseMessages[message] || "请求未完成，请重新加载历史记录后重试。"
    : message;
}

export function AnswerText({
  answer,
  sources,
  scope,
}: {
  answer: string;
  sources: ChatSource[];
  scope: string;
}) {
  const { t } = useLocale();
  return (
    <div className="answer-text">
      {answer.split(/(\[\d+\])/g).map((part, i) => {
        const match = /^\[(\d+)\]$/.exec(part);
        const source =
          match &&
          sources.find((item) => item.citation_id === Number(match[1]));
        return source ? (
          <a
            className="inline-citation"
            key={i}
            href={`#${scope}-source-${source.citation_id}`}
            aria-label={`${t("Source", "来源")} ${source.citation_id}: ${source.source}`}
          >
            {part}
          </a>
        ) : (
          part
        );
      })}
    </div>
  );
}

export function ChatTurn({
  turn,
  conversationId,
}: {
  turn: ConversationTurn;
  conversationId: string;
}) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const { t, formatDate } = useLocale();
  const answer = turn.response;
  const scope = `chat-${conversationId}-turn-${turn.sequence}`;
  return (
    <section
      className="chat-turn"
      aria-label={`${t("Question and answer", "问答")} ${turn.sequence}`}
    >
      <div className="user-question">
        <span className="eyebrow">{t("YOU", "你")}</span>
        <p>{turn.question}</p>
      </div>
      <div className="answer-card">
        <div className="answer-heading">
          <span className="answer-avatar">
            <Sparkles size={18} />
          </span>
          <strong>NewsLens</strong>
          <span className="answer-status">
            {answer.status === "answered"
              ? t("Grounded in reporting", "有新闻来源支持")
              : answer.status === "needs_clarification"
                ? t("Clarification needed", "需要进一步说明")
                : t("Not enough evidence", "证据不足")}
          </span>
          <button
            className="icon-button"
            aria-label={`${t("Copy answer", "复制回答")} ${turn.sequence}`}
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(answer.answer);
                setCopyError(false);
                setCopied(true);
              } catch {
                setCopyError(true);
              }
            }}
          >
            {copied ? <Check size={16} /> : <Copy size={16} />}
          </button>
        </div>
        <AnswerText
          answer={answer.answer}
          sources={answer.sources}
          scope={scope}
        />
        {answer.status === "insufficient_information" && (
          <div className="chat-guidance">
            <p>
              {answer.reason === "invalid_model_output"
                ? t(
                    "The answer could not be verified. Try a shorter question about one specific story.",
                    "暂时无法验证回答。请尝试针对一则具体新闻提出更简短的问题。",
                  )
                : t(
                    "This answer needs evidence from collected articles. Live weather and shopping prices are not connected. Try a specific story, publisher, or date.",
                    "回答需要已收录文章的支持。此聊天尚未接入实时天气或商品价格。请尝试指定新闻、媒体或日期。",
                  )}
            </p>
            <div>
              <Link href="/feed">
                {t("Browse available news", "浏览已收录新闻")}
              </Link>
              <Link href={`/search?q=${encodeURIComponent(turn.question)}`}>
                {t("Search this question", "搜索此问题")}
              </Link>
            </div>
          </div>
        )}
        {copied && (
          <p className="data-note" role="status">
            {t("Answer copied.", "已复制回答。")}
          </p>
        )}
        {copyError && (
          <p className="data-note" role="status">
            {t(
              "Copy is unavailable. Select the answer text to copy it.",
              "无法自动复制，请选中回答文字后复制。",
            )}
          </p>
        )}
        {!!answer.sources.length && (
          <div className="answer-sources">
            <h3>
              <BookOpen size={15} /> {t("Sources", "新闻来源")}{" "}
              <span>{answer.sources.length}</span>
            </h3>
            <div className="citation-grid">
              {answer.sources.map((source) => (
                <article
                  key={source.citation_id}
                  className="citation-card"
                  id={`${scope}-source-${source.citation_id}`}
                  tabIndex={-1}
                >
                  <div className="citation-meta">
                    <span>
                      [{source.citation_id}] {source.source}
                    </span>
                    <time dateTime={source.published_at || undefined}>
                      {formatDate(source.published_at)}
                    </time>
                  </div>
                  <SourceLink url={source.url}>{source.title}</SourceLink>
                  <details>
                    <summary>
                      {t("Read supporting excerpts", "查看原文依据")}
                    </summary>
                    {source.excerpts.map((excerpt, i) => (
                      <div key={i}>
                        <blockquote>{excerpt}</blockquote>
                        {source.translations?.[i] && (
                          <p className="source-translation">
                            <strong>{t("Translation", "译文")}</strong>{" "}
                            {source.translations[i]}
                          </p>
                        )}
                      </div>
                    ))}
                    <Link
                      className="collected-source-link"
                      href={`/articles/${source.article_id}`}
                    >
                      {t("Open collected article", "查看已收录文章")}{" "}
                      <ArrowRight size={13} />
                    </Link>
                  </details>
                </article>
              ))}
            </div>
          </div>
        )}
        {answer.follow_up && (
          <details className="context-disclosure">
            <summary>
              {t("How this follow-up was understood", "本次追问是如何理解的")}
            </summary>
            <p>{turn.resolved_question || answer.resolved_question}</p>
            <small>
              {t(
                "Earlier conversation helps locate reporting. The sources above support this answer.",
                "前面的对话用于查找相关报道。本次回答以上方列出的来源为依据。",
              )}
            </small>
          </details>
        )}
        <AnswerFreshness sources={answer.sources} />
        <div className="turn-saved">
          {t("Saved", "已保存")} {formatDate(turn.created_at, true)}
        </div>
      </div>
    </section>
  );
}

type ChatState = {
  info: ConversationInfo | null;
  turns: ConversationTurn[];
  nextBefore: number | null;
  question: string;
  mode: string;
  loading: boolean;
  pending: boolean;
  older: boolean;
  error: string;
  olderError: string;
  needsSync: boolean;
  loadFailed: boolean;
  note: string;
  recents: RecentConversation[];
  storageUnavailable: boolean;
  route: string;
  scroll: number;
};
const initial: ChatState = {
  info: null,
  turns: [],
  nextBefore: null,
  question: "",
  mode: "continue",
  loading: false,
  pending: false,
  older: false,
  error: "",
  olderError: "",
  needsSync: false,
  loadFailed: false,
  note: "",
  recents: [],
  storageUnavailable: false,
  route: "",
  scroll: 0,
};
type Patch = Partial<ChatState> | ((state: ChatState) => Partial<ChatState>);
function patchState(state: ChatState, patch: Patch): ChatState {
  return { ...state, ...(typeof patch === "function" ? patch(state) : patch) };
}
function failureMessage(error: unknown) {
  if (error instanceof ApiError) {
    if (error.status === 404)
      return "This conversation could not be found. Start a new conversation or choose another recent chat.";
    if (error.status === 409)
      return "This conversation changed elsewhere or reached its turn limit. Reload saved history before continuing.";
    return error.message;
  }
  return "The request could not be completed. Check the connection and reload saved history before sending again.";
}

export function Chat() {
  const { locale, t, formatDate } = useLocale();
  const params = useSearchParams();
  const router = useRouter();
  const id = params.get("conversation") || "";
  const seed = (params.get("q") || "").slice(0, 2000);
  const route = JSON.stringify([id, seed]);
  const [state, patch] = useReducer(patchState, {
    ...initial,
    route,
    question: seed,
    loading: !!id,
  });
  const epoch = useRef(0);
  const alive = useRef(true);
  const busy = useRef(false);
  const adoptedRoute = useRef<string | null>(null);
  const attempt = useRef<Attempt | null>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  useEffect(() => {
    if (adoptedRoute.current === route) {
      adoptedRoute.current = null;
      return;
    }
    adoptedRoute.current = null;
    const token = ++epoch.current;
    busy.current = false;
    attempt.current = null;
    let recents: RecentConversation[] = [];
    let storageUnavailable = false;
    try {
      recents = parseRecent(localStorage.getItem(recentKey));
    } catch {
      storageUnavailable = true;
    }
    patch({
      ...initial,
      route,
      question: seed,
      loading: !!id,
      recents,
      storageUnavailable,
    });
    if (!id) return;
    if (!conversationIdPattern.test(id)) {
      patch({
        loading: false,
        loadFailed: true,
        error: "This conversation link is invalid. Start a new conversation.",
      });
      return;
    }
    const controller = new AbortController();
    api<ConversationHistory>(`/conversations/${id}?limit=20`, {
      signal: controller.signal,
    }).then(
      (history) => {
        if (
          controller.signal.aborted ||
          token !== epoch.current ||
          !alive.current
        )
          return;
        patch({
          info: history.conversation,
          turns: history.turns,
          nextBefore: history.next_before,
          loading: false,
          scroll: 1,
        });
      },
      (error) => {
        if (
          !controller.signal.aborted &&
          token === epoch.current &&
          alive.current
        )
          patch({
            loading: false,
            loadFailed: true,
            error: failureMessage(error),
          });
      },
    );
    return () => controller.abort();
  }, [route, id, seed]);
  useEffect(() => {
    if (state.scroll || state.pending)
      end.current?.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
        block: "end",
      });
  }, [state.scroll, state.pending]);

  function remember(info: ConversationInfo, title: string) {
    let recents: RecentConversation[] = [];
    let unavailable = false;
    try {
      recents = rememberRecent(parseRecent(localStorage.getItem(recentKey)), {
        id: info.id,
        title,
        updatedAt: info.updated_at,
      });
      localStorage.setItem(recentKey, JSON.stringify(recents));
    } catch {
      unavailable = true;
    }
    patch((current) => ({
      recents: unavailable
        ? rememberRecent(current.recents, {
            id: info.id,
            title,
            updatedAt: info.updated_at,
          })
        : recents,
      storageUnavailable: unavailable,
    }));
  }
  async function send() {
    const question = state.question.trim();
    if (
      !question ||
      busy.current ||
      state.loading ||
      state.needsSync ||
      state.loadFailed ||
      (state.info?.revision || 0) >= 200
    )
      return;
    busy.current = true;
    const token = epoch.current;
    let info = state.info;
    patch({ pending: true, error: "", note: "" });
    try {
      if (!info) {
        info = await api<ConversationInfo>("/conversations", {
          method: "POST",
        });
        if (!alive.current || token !== epoch.current) return;
        remember(info, question);
        const nextRoute = JSON.stringify([info.id, ""]);
        adoptedRoute.current = nextRoute;
        patch({ info, route: nextRoute });
        router.replace(`/chat?conversation=${info.id}`, { scroll: false });
      }
      attempt.current = { id: info.id, revision: info.revision, question };
      const response = await api<ChatAnswer>("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          answer_language: locale,
          conversation_id: info.id,
          expected_revision: info.revision,
          follow_up: info.revision > 0 && state.mode === "continue",
          reset_context: state.mode === "new",
        }),
      });
      if (!alive.current || token !== epoch.current) return;
      if (
        response.conversation_id !== info.id ||
        response.conversation_revision !== info.revision + 1
      )
        throw new Error("Conversation response mismatch");
      const updated = {
        ...info,
        revision: response.conversation_revision,
        updated_at: new Date().toISOString(),
      };
      const turn: ConversationTurn = {
        sequence: updated.revision,
        question,
        resolved_question: response.resolved_question || question,
        response,
        created_at: updated.updated_at,
      };
      patch((current) => ({
        info: updated,
        turns: mergeTurns(current.turns, [turn]),
        question: "",
        mode: "continue",
        needsSync: false,
        scroll: current.scroll + 1,
      }));
      attempt.current = null;
      remember(updated, state.turns[0]?.question || question);
    } catch (error) {
      if (alive.current && token === epoch.current)
        patch({ error: failureMessage(error), needsSync: !!info });
    } finally {
      if (alive.current && token === epoch.current) {
        busy.current = false;
        patch({ pending: false });
        input.current?.focus();
      }
    }
  }
  async function reloadHistory() {
    const currentId = state.info?.id || id;
    if (!conversationIdPattern.test(currentId) || busy.current) return;
    busy.current = true;
    const token = epoch.current;
    patch({ loading: true, error: "", note: "" });
    try {
      const history = await api<ConversationHistory>(
        `/conversations/${currentId}?limit=20`,
      );
      if (!alive.current || token !== epoch.current) return;
      const recovered = recoveredAttempt(history, attempt.current);
      patch((current) => ({
        info: history.conversation,
        turns: history.turns,
        nextBefore: history.next_before,
        question: recovered ? "" : current.question,
        needsSync: false,
        loadFailed: false,
        note: recovered
          ? "Your saved answer was recovered. You can continue the conversation."
          : "Saved history refreshed. Review your draft before sending.",
        scroll: current.scroll + 1,
      }));
      attempt.current = null;
    } catch (error) {
      if (alive.current && token === epoch.current)
        patch({
          error: failureMessage(error),
          needsSync: true,
          loadFailed: !state.info,
        });
    } finally {
      if (alive.current && token === epoch.current) {
        busy.current = false;
        patch({ loading: false });
      }
    }
  }
  async function loadOlder() {
    if (!state.info || state.nextBefore === null || busy.current) return;
    busy.current = true;
    const token = epoch.current;
    patch({ older: true, olderError: "" });
    try {
      const history = await api<ConversationHistory>(
        `/conversations/${state.info.id}?limit=20&before=${state.nextBefore}`,
      );
      if (!alive.current || token !== epoch.current) return;
      if (history.conversation.revision !== state.info.revision) {
        patch({
          needsSync: true,
          error:
            "This conversation changed elsewhere. Reload saved history before continuing.",
        });
        return;
      }
      patch((current) => ({
        turns: mergeTurns(history.turns, current.turns),
        nextBefore: history.next_before,
      }));
    } catch (error) {
      if (alive.current && token === epoch.current)
        patch({ olderError: failureMessage(error) });
    } finally {
      if (alive.current && token === epoch.current) {
        busy.current = false;
        patch({ older: false });
      }
    }
  }
  function newChat() {
    if (busy.current) return;
    if (!id && !seed) {
      attempt.current = null;
      patch({ ...initial, route, recents: state.recents });
      input.current?.focus();
    } else router.push("/chat");
  }
  const navigating = state.route !== route;
  const locked =
    state.pending ||
    state.loading ||
    state.older ||
    state.needsSync ||
    state.loadFailed ||
    navigating;
  const full = (state.info?.revision || 0) >= 200;
  const workspaceInfo = (
    <>
      <div className="metro-collected">
        <BookOpen size={20} />
        <h2>{t("Collected news", "已收录新闻")}</h2>
        <p>
          {t(
            "Start with a story. Follow its sources.",
            "从一则新闻开始，追溯报道来源。",
          )}
        </p>
        <Link href="/feed">
          {t("View news feed", "浏览新闻列表")} <ArrowRight size={16} />
        </Link>
      </div>
      <div className="conversation-retention">
        <BookOpen size={17} />
        <p>
          {t(
            "Answers are saved in your local database. This browser remembers your 20 most recent chats.",
            "回答保存在本地数据库中。此浏览器会记住你最近的 20 个对话。",
          )}
        </p>
        {state.storageUnavailable && (
          <p role="status">
            {t(
              "Browser storage is unavailable. Keep the conversation URL to return to it.",
              "浏览器存储不可用。请保留对话网址，以便再次打开。",
            )}
          </p>
        )}
      </div>
    </>
  );
  return (
    <div className="conversation-page">
      <div className="conversation-layout">
        <aside className="conversation-rail">
          <button
            className="metro-new-chat"
            onClick={newChat}
            disabled={state.pending || state.older}
          >
            <Plus size={20} />
            {t("New chat", "新建对话")}
          </button>
          <div className="recent-title">
            <MessageSquare size={16} />
            <h2>{t("Recent chats", "最近对话")}</h2>
          </div>
          {state.recents.length ? (
            <nav aria-label={t("Recent conversations", "最近对话")}>
              {state.recents.map((recent) => (
                <button
                  key={recent.id}
                  className={recent.id === state.info?.id ? "selected" : ""}
                  disabled={state.pending || state.older}
                  onClick={() => router.push(`/chat?conversation=${recent.id}`)}
                >
                  <span>{recent.title}</span>
                  <small>{formatDate(recent.updatedAt)}</small>
                </button>
              ))}
            </nav>
          ) : (
            <p className="recent-empty">
              {t(
                "Your conversations will appear here after your first question.",
                "提出第一个问题后，对话将显示在这里。",
              )}
            </p>
          )}
          <div className="desktop-workspace-info">{workspaceInfo}</div>
        </aside>
        <div className="conversation-main">
          <div className="conversation-toolbar">
            <span>
              {state.info
                ? t(
                    `${state.info.revision} saved ${state.info.revision === 1 ? "answer" : "answers"}`,
                    `已保存 ${state.info.revision} 条回答`,
                  )
                : t("A fresh perspective", "开启新的视角")}
            </span>
            {(state.info || conversationIdPattern.test(id)) && (
              <button
                onClick={() => void reloadHistory()}
                disabled={state.pending || state.loading || state.older}
              >
                <RotateCcw size={13} />
                {t("Reload saved history", "重新加载历史记录")}
              </button>
            )}
          </div>
          {navigating || state.loading ? (
            <Loading label={t("Loading your conversation", "正在加载对话")} />
          ) : (
            <>
              {state.nextBefore !== null && (
                <button
                  className="older-messages"
                  disabled={state.older || state.pending}
                  onClick={() => void loadOlder()}
                >
                  <ChevronUp size={15} />
                  {state.older
                    ? t("Loading older messages…", "正在加载更早的消息…")
                    : t("Load older messages", "加载更早的消息")}
                </button>
              )}
              {state.olderError && (
                <p role="alert" className="form-error">
                  {localizeMessage(state.olderError, locale)}{" "}
                  {t(
                    "Use “Load older messages” to retry.",
                    "点击“加载更早的消息”重试。",
                  )}
                </p>
              )}
              {!state.turns.length && !state.pending && !state.loadFailed && (
                <div className="chat-welcome">
                  <span className="welcome-symbol">
                    <Sparkles size={31} />
                  </span>
                  <h2>{t("Start with the sources.", "从新闻来源开始。")}</h2>
                  <p>
                    {t(
                      "Start with a story. Follow up with another question.",
                      "从一则新闻开始，随时继续追问。",
                    )}
                  </p>
                  <p className="data-note">
                    {t(
                      "Choose a story from the news feed, then ask about its details. Answers are limited to the articles in your library.",
                      "先从新闻列表选择一则报道，再询问其中的细节。回答范围限于新闻库中已收录的文章。",
                    )}
                  </p>
                  <div className="chat-suggestions">
                    {[
                      t(
                        "What is the Xbox console limit?",
                        "Xbox 游戏机有哪些限制？",
                      ),
                      t(
                        "What did OpenAI claim about solving a maths problem?",
                        "OpenAI 对解决数学问题作出了什么声明？",
                      ),
                      t(
                        "What are the latest developments in Malaysia?",
                        "马来西亚最近有哪些新动态？",
                      ),
                    ].map((suggestion) => (
                      <button
                        key={suggestion}
                        onClick={() => {
                          patch({ question: suggestion });
                          input.current?.focus();
                        }}
                      >
                        <span>{suggestion}</span>
                        <ArrowRight size={15} />
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div
                className="conversation-messages"
                aria-label={t("Conversation messages", "对话消息")}
              >
                {state.turns.map((turn) => (
                  <ChatTurn
                    key={turn.sequence}
                    turn={turn}
                    conversationId={state.info!.id}
                  />
                ))}
              </div>
              {state.pending && (
                <>
                  <div className="user-question pending-question">
                    <span className="eyebrow">{t("YOU", "你")}</span>
                    <p>{state.question.trim()}</p>
                  </div>
                  <div className="answer-pending" role="status">
                    <span className="spinner" />
                    <div>
                      <strong>
                        {t("Reading the coverage…", "正在阅读相关报道…")}
                      </strong>
                      <p>
                        {t(
                          "Finding evidence and checking sources. Local models can take a minute or two.",
                          "正在查找依据并核对来源，本地模型可能需要一两分钟。",
                        )}
                      </p>
                    </div>
                  </div>
                </>
              )}
            </>
          )}
          {state.error && (
            <div className="conversation-error" role="alert">
              <strong>
                {t(
                  "We couldn’t finish that request.",
                  "暂时无法完成此次请求。",
                )}
              </strong>
              <p>{localizeMessage(state.error, locale)}</p>
              {state.needsSync && (
                <p>
                  {t(
                    "Your draft is preserved. Reload saved history before sending again; the request may already have been saved.",
                    "你的草稿已保留。请先重新加载历史记录再发送，因为此次请求可能已保存。",
                  )}
                </p>
              )}
              {(state.info || conversationIdPattern.test(id)) && (
                <button
                  className="button secondary small"
                  disabled={state.loading || state.pending}
                  onClick={() => void reloadHistory()}
                >
                  {t("Reload history to recover", "重新加载历史记录以恢复")}
                </button>
              )}
            </div>
          )}
          {state.note && (
            <p className="conversation-note" role="status">
              {localizeMessage(state.note, locale)}
            </p>
          )}
          {full && (
            <p className="conversation-note" role="status">
              {t(
                "This conversation has reached 200 turns. Start a new chat to continue.",
                "此对话已达到 200 轮。请新建对话以继续。",
              )}
            </p>
          )}
          <div ref={end} />
          <form
            className="chat-composer conversation-composer"
            onSubmit={(event) => {
              event.preventDefault();
              void send();
            }}
          >
            <label className="sr-only" htmlFor="chat-question">
              {t("Your question", "你的问题")}
            </label>
            <textarea
              ref={input}
              id="chat-question"
              value={state.question}
              onChange={(event) => patch({ question: event.target.value })}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey &&
                  !event.nativeEvent.isComposing
                ) {
                  event.preventDefault();
                  if (!locked && !full) void send();
                }
              }}
              placeholder={
                state.info?.revision
                  ? t("Ask a follow-up question…", "继续追问…")
                  : t(
                      "Ask about a story, a topic, or a source…",
                      "询问一则新闻、一个话题或一家媒体…",
                    )
              }
              maxLength={2000}
              rows={3}
              required
              disabled={locked || full}
            />
            <div className="composer-bottom">
              <span className="composer-language">
                {t("Replies in English", "将用中文回复")}
              </span>
              {!!state.info?.revision && (
                <label className="context-mode">
                  <span className="sr-only">
                    {t("Question context", "问题上下文")}
                  </span>
                  <select
                    aria-label={t("Question context", "问题上下文")}
                    value={state.mode}
                    onChange={(event) => patch({ mode: event.target.value })}
                    disabled={locked}
                  >
                    <option value="continue">
                      {t("Follow-up", "继续追问")}
                    </option>
                    <option value="new">{t("New topic", "新话题")}</option>
                  </select>
                </label>
              )}
              <span>{state.question.length}/2000</span>
              <button
                type="submit"
                className="send-button"
                disabled={locked || full || !state.question.trim()}
                aria-label={t("Send question", "发送问题")}
              >
                <ArrowRight size={21} />
              </button>
            </div>
          </form>
          <p className="chat-footnote">
            {t(
              "Enter to send · Shift+Enter for a new line. Follow-ups use recent context and fresh source retrieval. Answers use collected news, not the live web.",
              "Enter 发送 · Shift+Enter 换行。追问会结合最近的对话并重新检索来源。回答依据已收录的新闻，不会实时浏览互联网。",
            )}
          </p>
        </div>
        <details className="mobile-workspace-info">
          <summary>{t("About this workspace", "关于此工作区")}</summary>
          {workspaceInfo}
        </details>
      </div>
    </div>
  );
}
