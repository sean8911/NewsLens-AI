/** @jsxImportSource react */
"use client";

import Link from "next/link";
import {
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  RotateCcw,
  Rss,
  Search,
  Sparkles,
} from "lucide-react";
import type { ReactNode } from "react";
import { date, safeUrl } from "@/lib/api";
import { useLocale } from "./locale";
import type { Article, NewsEvent } from "@/lib/types";

export function Header({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action}
    </div>
  );
}

export function Loading({ label = "Loading your news" }: { label?: string }) {
  return (
    <div className="loading-block" role="status">
      <span className="spinner" />
      {label}
      <div className="skeleton-grid" aria-hidden="true">
        {[0, 1, 2].map((n) => (
          <div key={n} className="skeleton-card">
            <i />
            <i />
            <i />
          </div>
        ))}
      </div>
    </div>
  );
}

export function Empty({
  title,
  children,
  action,
}: {
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <Search size={24} />
      </div>
      <h2>{title}</h2>
      <p>{children}</p>
      {action}
    </div>
  );
}

export function ErrorState({
  message,
  retry,
}: {
  message: string;
  retry?: () => void;
}) {
  return (
    <div className="error-state" role="alert">
      <div>
        <strong>We couldn’t load this view.</strong>
        <p>{message}</p>
      </div>
      {retry && (
        <button className="button secondary" onClick={retry}>
          <RotateCcw size={15} />
          Try again
        </button>
      )}
    </div>
  );
}

export function SourceLink({
  url,
  children,
  className = "source-link",
}: {
  url: string;
  children: ReactNode;
  className?: string;
}) {
  const { t } = useLocale();
  const href = safeUrl(url);
  return href ? (
    <a
      className={className}
      href={href}
      target="_blank"
      rel="noopener noreferrer"
    >
      {children}
      <ArrowUpRight size={14} />
      <span className="sr-only">
        {" "}
        {t("(opens in a new tab)", "（在新标签页中打开）")}
      </span>
    </a>
  ) : (
    <span className={className}>{children}</span>
  );
}

export function ArticleCard({
  article,
  feature = false,
}: {
  article: Article;
  feature?: boolean;
}) {
  const topic = article.article_topics[0]?.topic || article.category;
  return (
    <article className={`article-card ${feature ? "feature-story" : ""}`}>
      <div className="story-topline">
        <Link
          className="topic-label"
          href={`/category/${encodeURIComponent(topic)}?kind=${article.article_topics.length ? "topic" : "category"}`}
        >
          {topic}
        </Link>
        <span className="publisher-mark">{article.source.slice(0, 1)}</span>
      </div>
      <h2>
        <Link href={`/articles/${article.id}`}>{article.title}</Link>
      </h2>
      <p className="story-summary">
        {article.summary ||
          article.description ||
          "Open this article to explore the original reporting."}
      </p>
      {!!article.entities.length && (
        <div className="entity-tags" aria-label="Important entities">
          {article.entities.slice(0, 3).map((entity) => (
            <Link key={entity.id} href={`/feed?entity_id=${entity.id}`}>
              {entity.name}
            </Link>
          ))}
        </div>
      )}
      <div className="story-footer">
        <div>
          <span className="source-name">
            <Rss size={12} />
            {article.source}
          </span>
          <span className="story-date">
            1 source · Reported {date(article.published_at, true)}
          </span>
        </div>
        <Link
          href={`/articles/${article.id}`}
          className="round-arrow"
          aria-label={`Read ${article.title}`}
        >
          <ArrowUpRight size={19} />
        </Link>
      </div>
    </article>
  );
}

export function EventCard({ event }: { event: NewsEvent }) {
  return (
    <article className="article-card event-card">
      <div className="story-topline">
        <span className="topic-label">Connected coverage</span>
        <span className="tiny-count">{event.article_count} articles</span>
      </div>
      <h2>
        <Link href={`/events/${event.id}`}>{event.title}</Link>
      </h2>
      <p className="story-summary">
        {!event.stale && !event.event_summary.stale && event.summary
          ? event.event_summary.document?.what_happened.text || event.summary
          : "Explore the reporting behind this developing story."}
      </p>
      <div className="story-footer">
        <div>
          <span className="source-name">
            <BookOpen size={13} />
            {event.source_count}{" "}
            {event.source_count === 1 ? "source" : "sources"}
          </span>
          <span className="story-date">
            Updated {date(event.last_updated_at, true)}
          </span>
        </div>
        <Link
          href={`/events/${event.id}`}
          className="round-arrow"
          aria-label={`Explore ${event.title}`}
        >
          <ArrowRight size={18} />
        </Link>
      </div>
    </article>
  );
}

export function Pager({
  offset,
  limit,
  total,
  hasMore,
  href,
}: {
  offset: number;
  limit: number;
  total: number;
  hasMore: boolean;
  href: (offset: number) => string;
}) {
  return (
    <nav className="pagination" aria-label="Pagination">
      <span>
        {total
          ? `${offset + 1}–${Math.min(offset + limit, total)} of ${total} results`
          : "0 results"}
      </span>
      <div>
        {offset > 0 ? (
          <Link
            className="button secondary"
            href={href(Math.max(0, offset - limit))}
          >
            <ChevronLeft size={16} />
            Previous
          </Link>
        ) : (
          <button className="button secondary" disabled>
            <ChevronLeft size={16} />
            Previous
          </button>
        )}
        {hasMore ? (
          <Link className="button secondary" href={href(offset + limit)}>
            Next
            <ChevronRight size={16} />
          </Link>
        ) : (
          <button className="button secondary" disabled>
            Next
            <ChevronRight size={16} />
          </button>
        )}
      </div>
    </nav>
  );
}

export function AskBanner() {
  return (
    <div className="ask-banner">
      <div className="ask-icon">
        <Sparkles size={23} />
      </div>
      <div>
        <h2>Go beyond the headline.</h2>
        <p>Ask a question. Follow the evidence.</p>
      </div>
      <Link className="button dark" href="/chat">
        Ask NewsLens
        <ArrowUpRight size={16} />
      </Link>
    </div>
  );
}
