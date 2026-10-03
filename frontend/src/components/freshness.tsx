/** @jsxImportSource react */
"use client";

import { useEffect, useReducer } from "react";
import { Clock3, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { useLocale } from "./locale";
import type { ChatSource, Status } from "@/lib/types";
import { relativeTime, timestamp, updateLabel } from "@/lib/freshness";

type FreshnessState = {
  data?: Status;
  error: boolean;
  loading: boolean;
  receivedAt: number;
  clock: number;
};
export function useFreshness() {
  const [revision, refresh] = useReducer((n: number) => n + 1, 0);
  const [state, patch] = useReducer(
    (s: FreshnessState, p: Partial<FreshnessState>) => ({ ...s, ...p }),
    { error: false, loading: true, receivedAt: 0, clock: 0 },
  );
  useEffect(() => {
    let disposed = false;
    let controller: AbortController | undefined;
    async function load() {
      if (controller) return;
      controller = new AbortController();
      patch({ loading: true });
      try {
        const data = await api<Status>("/system/status", {
          signal: controller.signal,
        });
        if (!disposed) {
          const clock = performance.now();
          patch({ data, error: false, receivedAt: clock, clock });
        }
      } catch {
        if (!disposed) patch({ error: true });
      } finally {
        controller = undefined;
        if (!disposed) patch({ loading: false });
      }
    }
    void load();
    const poll = window.setInterval(() => {
      if (document.visibilityState === "visible") void load();
    }, 60_000);
    const tick = window.setInterval(
      () => patch({ clock: performance.now() }),
      30_000,
    );
    const focus = () => {
      if (document.visibilityState === "visible") void load();
    };
    window.addEventListener("focus", focus);
    document.addEventListener("visibilitychange", focus);
    return () => {
      disposed = true;
      controller?.abort();
      window.clearInterval(poll);
      window.clearInterval(tick);
      window.removeEventListener("focus", focus);
      document.removeEventListener("visibilitychange", focus);
    };
  }, [revision]);
  const checked = timestamp(state.data?.checked_at);
  return {
    ...state,
    now:
      checked === null
        ? NaN
        : checked + Math.max(0, state.clock - state.receivedAt),
    refresh,
  };
}

function ExactTime({
  value,
  fallback,
}: {
  value: string | null | undefined;
  fallback?: string;
}) {
  const { t, formatDate } = useLocale();
  return timestamp(value) === null ? (
    <span>{fallback || t("Not recorded", "暂无记录")}</span>
  ) : (
    <time dateTime={value!}>{formatDate(value!, true)}</time>
  );
}

export function FreshnessPanel({
  data,
  error,
  loading,
  now,
  refresh,
}: {
  data?: Status;
  error: boolean;
  loading: boolean;
  now: number;
  refresh: () => void;
}) {
  const { locale, t } = useLocale();
  const attemptStatus = (value: string | null | undefined) => {
    const labels: Record<string, string> = {
      failed: "失败",
      interrupted: "已中断",
      running: "进行中",
      partial: "部分完成",
      succeeded: "成功",
      skipped: "已跳过",
    };
    return value
      ? locale === "zh"
        ? labels[value] || "未知状态"
        : value
      : t("not recorded", "暂无记录");
  };
  const available = data?.status === "ok" && data.database === "available";
  const unavailable = error || (!!data && !available);
  const label = unavailable
    ? t("Freshness unavailable", "无法获取新闻更新状态")
    : !data
      ? t("Checking collection history…", "正在检查采集记录…")
      : updateLabel(data.last_successful_news_update, now, locale);
  return (
    <section
      className={`freshness-panel ${unavailable ? "freshness-unavailable" : ""}`}
      aria-label={t("News freshness", "新闻更新状态")}
    >
      <div className="freshness-heading">
        <span>
          <Clock3 size={15} />
          <strong>{label}</strong>
        </span>
        <button
          onClick={refresh}
          disabled={loading}
          aria-label={t("Refresh freshness status", "刷新新闻更新状态")}
        >
          <RefreshCw size={13} />
          {loading
            ? t("Checking…", "正在检查…")
            : t("Refresh status", "刷新状态")}
        </button>
      </div>
      {unavailable && (
        <p className="freshness-warning" role="status">
          {error && available
            ? t(
                "Status check failed. Times below are last known values.",
                "状态检查失败，下方显示的是上次记录的时间。",
              )
            : t(
                "The database could not be checked. News coverage cannot be confirmed.",
                "无法连接数据库，暂时不能确认新闻覆盖情况。",
              )}
        </p>
      )}
      {available &&
        data.last_update_attempt_status &&
        ["failed", "interrupted", "running"].includes(
          data.last_update_attempt_status,
        ) && (
          <p className="freshness-warning">
            {t("Latest update attempt:", "最近一次更新尝试：")}{" "}
            {attemptStatus(data.last_update_attempt_status)}.{" "}
            {t(
              "The last successful update shown below has not advanced.",
              "下方显示的最近成功更新时间尚未推进。",
            )}
          </p>
        )}
      {(available || (!data && !error)) && (
        <div className="freshness-facts">
          <div>
            <span>{t("Last news update", "最近新闻更新")}</span>
            <strong>
              {data
                ? relativeTime(data.last_successful_news_update, now, locale)
                : t("Checking…", "正在检查…")}
            </strong>
            <small>
              <ExactTime value={data?.last_successful_news_update} />
              {data?.last_successful_update_status === "partial" &&
                t(" · Partial update", " · 部分更新")}
            </small>
          </div>
          <div>
            <span>
              {t("Latest article · published", "最新文章 · 发布时间")}
            </span>
            <strong>
              <ExactTime
                value={data?.latest_article_published_at}
                fallback={
                  data?.article_count === 0
                    ? t("No articles collected", "尚未收录文章")
                    : t("Publication date unknown", "发布日期未知")
                }
              />
            </strong>
            <small>
              {timestamp(data?.latest_article_published_at)! > now
                ? t(
                    "Future publication timestamp — verify the source date.",
                    "发布时间晚于当前时间，请核对来源日期。",
                  )
                : t(
                    "Publication time is different from collection time.",
                    "发布时间与采集时间不同。",
                  )}
            </small>
          </div>
        </div>
      )}
      <details>
        <summary>{t("About this coverage", "关于新闻覆盖范围")}</summary>
        <p>
          {t(
            "Answers use collected articles, not the live web. A recent collection check does not guarantee recent reporting or complete coverage.",
            "回答依据已收录的文章，不会实时浏览互联网。最近检查过采集状态，不代表报道足够新或覆盖完整。",
          )}
        </p>
        {available && (
          <>
            <p>
              {t("Most recent article collection:", "最近文章采集时间：")}{" "}
              <ExactTime value={data.latest_article_retrieved_at} />.
            </p>
            <p>
              {t("Latest update attempt:", "最近一次更新尝试：")}{" "}
              {attemptStatus(data.last_update_attempt_status)} ·{" "}
              <ExactTime value={data.last_update_attempt_at} />.{" "}
              {t(
                "Partial updates mean some feeds failed. Failed, interrupted or running attempts do not advance the last successful update.",
                "部分更新表示有些新闻源采集失败。失败、已中断或进行中的尝试不会推进最近成功更新时间。",
              )}
            </p>
            <p>
              {t("Status checked:", "状态检查时间：")}{" "}
              <ExactTime value={data.checked_at} />.{" "}
              {t(
                "Status refreshes every minute while this page is visible; this does not collect news. “Update overdue” means no successful update in 30 minutes.",
                "页面可见时，每分钟刷新一次状态，但不会采集新闻。“新闻待更新”表示已超过 30 分钟没有成功更新。",
              )}
            </p>
          </>
        )}
      </details>
    </section>
  );
}

export function AnswerFreshness({ sources }: { sources: ChatSource[] }) {
  const { t } = useLocale();
  if (!sources.length)
    return (
      <p className="answer-freshness">
        {t(
          "No cited sources; a collection cutoff is unavailable.",
          "没有引用来源，无法确定采集截止时间。",
        )}
      </p>
    );
  const collected = sources.map((source) => timestamp(source.retrieved_at));
  const dated = sources
    .map((source) => timestamp(source.published_at))
    .filter((value): value is number => value !== null);
  const complete = collected.every((value): value is number => value !== null);
  return (
    <div className="answer-freshness">
      <p>
        {complete ? (
          <>
            {t(
              "Based on cited articles collected up to",
              "本回答依据的引用文章采集于以下时间或之前：",
            )}{" "}
            <ExactTime value={new Date(Math.max(...collected)).toISOString()} />
            .
          </>
        ) : (
          t(
            "Collection time is not recorded for every source in this saved answer.",
            "此历史回答的部分来源未记录采集时间。",
          )
        )}
      </p>
      <p>
        {dated.length ? (
          <>
            {t("Latest dated source:", "来源中的最新发布日期：")}{" "}
            <ExactTime value={new Date(Math.max(...dated)).toISOString()} />
            {dated.length < sources.length
              ? t(
                  " (some publication dates are unknown)",
                  "（部分来源的发布日期未知）",
                )
              : ""}
            .
          </>
        ) : (
          t("Source publication dates are unknown.", "来源的发布日期未知。")
        )}{" "}
        {t(
          "This is the evidence for this answer, not a claim to cover newer news.",
          "以上信息说明本次回答的依据，不代表已覆盖更新的新闻。",
        )}
      </p>
    </div>
  );
}
