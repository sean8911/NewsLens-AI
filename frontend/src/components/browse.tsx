"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import { date, query, useApi } from "@/lib/api";
import type { Article, EventPage, Page, Trends } from "@/lib/types";
import {
  ArticleCard,
  Empty,
  ErrorState,
  EventCard,
  Header,
  Loading,
  Pager,
  SourceLink,
} from "./ui";

function pageOffset(value: string | null) {
  const parsed = Number(value || 0);
  return Number.isInteger(parsed) && parsed >= 0 && parsed <= 1_000_000
    ? parsed
    : 0;
}

export function Feed({ category }: { category?: string }) {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const [filterError, setFilterError] = useState("");
  const offset = pageOffset(params.get("offset"));
  const kind = params.get("kind") === "topic" ? "topic" : "category";
  const values: Record<string, string | number | undefined> = {
    limit: 12,
    offset,
    category:
      category && kind === "category"
        ? category
        : params.get("category") || undefined,
    topic:
      category && kind === "topic"
        ? category === "AI"
          ? "Artificial Intelligence"
          : category
        : params.get("topic") || undefined,
    source: params.get("source") || undefined,
    language: params.get("language") || undefined,
    entity_id: params.get("entity_id") || undefined,
    date_from: params.get("date_from") || undefined,
    date_to: params.get("date_to") || undefined,
  };
  const result = useApi<Page<Article>>(`/articles?${query(values)}`);
  const href = (nextOffset: number) => {
    const next = new URLSearchParams(params.toString());
    next.set("offset", String(nextOffset));
    return `${pathname}?${next}`;
  };
  return (
    <>
      <Header
        eyebrow={category ? "TOPIC COLLECTION" : "THE READING ROOM"}
        title={category || "Your news feed."}
        description={
          category
            ? `The latest collected reporting on ${category}. Follow the details that matter to you.`
            : "Original reporting, a little more organized. Find your next perspective."
        }
      />
      <form
        className="filter-bar"
        key={params.toString()}
        onSubmit={(event) => {
          event.preventDefault();
          const data = new FormData(event.currentTarget);
          const next = new URLSearchParams();
          const from = String(data.get("date_from") || ""),
            to = String(data.get("date_to") || "");
          if (from && to && from > to) {
            setFilterError("The start date must be on or before the end date.");
            return;
          }
          setFilterError("");
          data.forEach((value, key) => {
            if (String(value).trim()) next.set(key, String(value).trim());
          });
          if (category) next.set("kind", kind);
          if (params.has("entity_id"))
            next.set("entity_id", params.get("entity_id")!);
          if (!category && params.has("topic"))
            next.set("topic", params.get("topic")!);
          router.push(`${pathname}?${next}`);
        }}
      >
        <span className="filter-icon">
          <SlidersHorizontal size={18} />
          Refine
        </span>
        {!category && (
          <label>
            Category
            <input
              name="category"
              defaultValue={String(values.category || "")}
              placeholder="Any category"
              maxLength={100}
            />
          </label>
        )}
        <label>
          Source
          <input
            name="source"
            defaultValue={String(values.source || "")}
            placeholder="Any publisher"
            maxLength={200}
          />
        </label>
        <label>
          Language
          <select name="language" defaultValue={String(values.language || "")}>
            <option value="">All languages</option>
            <option value="en">English</option>
            <option value="ms">Malay</option>
            <option value="zh">Chinese</option>
            <option value="unknown">Unidentified</option>
          </select>
        </label>
        <label>
          From
          <input
            type="date"
            name="date_from"
            defaultValue={String(values.date_from || "")}
          />
        </label>
        <label>
          Until
          <input
            type="date"
            name="date_to"
            defaultValue={String(values.date_to || "")}
          />
        </label>
        <button className="button" type="submit">
          Apply
        </button>
        <Link
          className="clear-filter"
          onClick={() => setFilterError("")}
          href={category ? `${pathname}?kind=${kind}` : "/feed"}
        >
          Clear
        </Link>
      </form>
      {filterError && (
        <p role="alert" className="form-error">
          {filterError}
        </p>
      )}
      <div className="results-topline">
        <span>
          {result.data
            ? `${result.data.total} stories${values.topic ? ` · ${values.topic}` : ""}`
            : "Your collection"}
        </span>
        <span>Newest publication first</span>
      </div>
      {result.loading ? (
        <Loading />
      ) : result.error ? (
        <ErrorState message={result.error} retry={result.retry} />
      ) : (
        result.data && (
          <>
            {result.data.count ? (
              <div className="article-grid">
                {result.data.results.map((article) => (
                  <ArticleCard key={article.id} article={article} />
                ))}
              </div>
            ) : (
              <Empty
                title="No stories in this view."
                action={
                  <Link className="button secondary" href="/feed">
                    Browse all stories
                    <ArrowRight size={15} />
                  </Link>
                }
              >
                Try a different source, date range, or category.
              </Empty>
            )}
            <Pager
              offset={offset}
              limit={12}
              total={result.data.total}
              hasMore={result.data.has_more}
              href={href}
            />
          </>
        )
      )}
    </>
  );
}

export function ArticlePage({ id }: { id: string }) {
  const result = useApi<Article>(`/articles/${id}`);
  if (result.loading) return <Loading label="Opening the story" />;
  if (result.error)
    return <ErrorState message={result.error} retry={result.retry} />;
  const article = result.data!;
  return (
    <>
      <Link className="back-link" href="/feed">
        <ArrowLeft size={15} />
        Back to the news feed
      </Link>
      <div className="reading-layout">
        <article className="reading-paper">
          <div className="eyebrow">
            {article.category} · {article.source}
          </div>
          <h1>{article.title}</h1>
          <div className="reading-meta">
            <span>{date(article.published_at)}</span>
            {article.author && <span>By {article.author}</span>}
            <span>1 original source</span>
          </div>
          {article.summary && (
            <div className="summary-box">
              <span className="eyebrow">
                <Sparkles size={14} />
                AT A GLANCE
              </span>
              <p>{article.summary}</p>
            </div>
          )}
          <div className="article-body">
            {(
              article.content ||
              article.description ||
              "The full article text is not available in this collection. Read the original reporting below."
            )
              .split(/\n+/)
              .filter(Boolean)
              .map((paragraph, i) => (
                <p key={i}>{paragraph}</p>
              ))}
          </div>
          <div className="original-source">
            <span>Continue with the original reporting</span>
            <SourceLink url={article.url}>{article.source}</SourceLink>
          </div>
        </article>
        <aside className="reading-aside">
          <div className="topics-box">
            <span className="eyebrow">IN THIS STORY</span>
            <h3>People, places & ideas.</h3>
            <div className="entity-tags">
              {article.entities.map((entity) => (
                <Link key={entity.id} href={`/feed?entity_id=${entity.id}`}>
                  {entity.name}
                </Link>
              ))}
            </div>
            {!article.entities.length && (
              <p className="muted-copy">
                No entities identified for this article.
              </p>
            )}
          </div>
          <div className="mini-ask">
            <Sparkles size={24} />
            <h3>Put it in perspective.</h3>
            <p>
              Explore what your collected reporting says about this subject.
            </p>
            <Link
              className="button"
              href={`/chat?q=${encodeURIComponent(`What does the reporting say about ${article.title}?`.slice(0, 2000))}`}
            >
              Ask about this story
              <ArrowUpRight size={14} />
            </Link>
          </div>
        </aside>
      </div>
    </>
  );
}

export function Events() {
  const params = useSearchParams();
  const offset = pageOffset(params.get("offset"));
  const result = useApi<EventPage>(`/events?limit=9&offset=${offset}`);
  return (
    <>
      <Header
        eyebrow="CONNECTED REPORTING"
        title="One story. Many perspectives."
        description="See how individual reports come together around a shared event."
      />
      {result.loading ? (
        <Loading label="Connecting the coverage" />
      ) : result.error ? (
        <ErrorState message={result.error} retry={result.retry} />
      ) : (
        result.data && (
          <>
            {result.data.count ? (
              <div className="article-grid">
                {result.data.results.map((event) => (
                  <EventCard key={event.id} event={event} />
                ))}
              </div>
            ) : (
              <Empty
                title="Stories are still coming together."
                action={
                  <Link className="button secondary" href="/feed">
                    Explore individual articles
                    <ArrowRight size={15} />
                  </Link>
                }
              >
                There are no grouped news events in this collection yet.
              </Empty>
            )}
            <Pager
              offset={offset}
              limit={9}
              total={result.data.total}
              hasMore={offset + result.data.count < result.data.total}
              href={(next) => `/events?offset=${next}`}
            />
          </>
        )
      )}
    </>
  );
}

export function Trending() {
  const params = useSearchParams();
  const router = useRouter();
  const hours = [24, 72, 168].includes(Number(params.get("hours")))
    ? Number(params.get("hours"))
    : 24;
  const result = useApi<Trends>(`/trending?window_hours=${hours}&limit=12`);
  return (
    <>
      <Header
        eyebrow="THE ATTENTION MAP"
        title="What’s gaining momentum."
        description="Multi-source coverage, measured across the reporting in your library."
      />
      <div className="trend-toolbar">
        <div className="segmented" aria-label="Trending time window">
          {[
            [24, "24 hours"],
            [72, "3 days"],
            [168, "7 days"],
          ].map(([value, label]) => (
            <button
              key={value}
              aria-pressed={hours === value}
              onClick={() => router.push(`/trending?hours=${value}`)}
            >
              {label}
            </button>
          ))}
        </div>
        <span>At least 2 independent sources</span>
      </div>
      {result.loading ? (
        <Loading label="Looking across the sources" />
      ) : result.error ? (
        <ErrorState message={result.error} retry={result.retry} />
      ) : (
        result.data && (
          <>
            {!result.data.count ? (
              <Empty
                title="A quieter window, not the whole picture."
                action={
                  <Link href="/feed" className="button secondary">
                    Explore your library
                    <ArrowRight size={15} />
                  </Link>
                }
              >
                No topics meet the multi-source threshold in this window. Try 7
                days or explore the full collection.
              </Empty>
            ) : (
              <div className="trends-grid">
                {result.data.results.map((trend, i) => (
                  <article className="trend-card" key={trend.id}>
                    <div className="story-topline">
                      <span className="trend-rank">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <span className="trend-state">
                        {trend.state.replaceAll("_", " ")}
                      </span>
                    </div>
                    <h2>
                      <Link
                        href={`/category/${encodeURIComponent(trend.topic)}?kind=topic`}
                      >
                        {trend.label}
                      </Link>
                    </h2>
                    <div className="trend-numbers">
                      <span>
                        <strong>{trend.article_count}</strong> articles
                      </span>
                      <span>
                        <strong>{trend.source_count}</strong> sources
                      </span>
                    </div>
                    <div className="trend-source-list">
                      {trend.sources.join(" · ")}
                    </div>
                    {trend.evidence.slice(0, 2).map((evidence) => (
                      <Link
                        className="trend-evidence"
                        key={evidence.article_id}
                        href={`/articles/${evidence.article_id}`}
                      >
                        {evidence.title}
                        <ArrowUpRight size={14} />
                      </Link>
                    ))}
                  </article>
                ))}
              </div>
            )}
            <p className="data-note">
              Window ends {date(result.data.as_of, true)}. Based on publication
              times in your collected news; this is not a measure of all news
              worldwide.
            </p>
          </>
        )
      )}
    </>
  );
}
