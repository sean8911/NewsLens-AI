/** @jsxImportSource react */
"use client";

import Link from "next/link";
import { useState } from "react";
import {
  ArrowLeft,
  ArrowUpRight,
  BookOpen,
  Clock3,
  Layers3,
  Sparkles,
  GitCompareArrows,
  ListChecks,
  Users,
  ExternalLink,
} from "lucide-react";
import { date, useApi } from "@/lib/api";
import type {
  CitedFact,
  EvidenceCitation,
  NewsEvent,
  EventComparison,
  EventTimeline,
  TimelineItem,
} from "@/lib/types";
import { ErrorState, Loading, SourceLink } from "./ui";

function Notice({ children }: { children: React.ReactNode }) {
  return <p className="intelligence-notice">{children}</p>;
}

export function CitedText({
  fact,
  citations,
  scope,
}: {
  fact: CitedFact;
  citations: EvidenceCitation[];
  scope: string;
}) {
  return (
    <div className="cited-fact">
      <p>{fact.text}</p>
      <div className="fact-citations">
        {fact.citation_ids.map((id) => {
          const citation = citations.find((item) => item.citation_id === id);
          return citation ? (
            <a
              key={id}
              href={`#${scope}-citation-${id}`}
              aria-label={`View ${scope} evidence ${id} from ${citation.source}`}
            >
              [{id}] {citation.source}
            </a>
          ) : (
            <span key={id}>[{id}] Evidence unavailable</span>
          );
        })}
      </div>
    </div>
  );
}

export function Evidence({
  citations,
  scope,
}: {
  citations: EvidenceCitation[];
  scope: string;
}) {
  if (!citations.length) return null;
  return (
    <div className="evidence-register">
      <h3>
        <BookOpen size={16} /> {scope === "summary" ? "Summary" : "Comparison"}{" "}
        evidence
      </h3>
      <div className="evidence-grid">
        {citations.map((citation) => (
          <article
            className="evidence-entry"
            id={`${scope}-citation-${citation.citation_id}`}
            key={citation.citation_id}
            tabIndex={-1}
          >
            <div className="evidence-label">
              [{citation.citation_id}] {citation.source}{" "}
              <time dateTime={citation.published_at || undefined}>
                {date(citation.published_at)}
              </time>
            </div>
            <SourceLink url={citation.url}>{citation.title}</SourceLink>
            <details>
              <summary>Read supporting excerpt</summary>
              <blockquote>{citation.excerpt}</blockquote>
              <Link href={`/articles/${citation.article_id}`}>
                Open collected article <ArrowUpRight size={12} />
              </Link>
            </details>
          </article>
        ))}
      </div>
    </div>
  );
}

export function SummarySection({ event }: { event: NewsEvent }) {
  const view = event.event_summary;
  const stale = event.stale || view.stale || view.status === "stale";
  const doc = !stale && view.status === "summarized" ? view.document : null;
  return (
    <section
      id="event-summary"
      className="intelligence-section"
      aria-labelledby="summary-heading"
    >
      <div className="intelligence-section-heading">
        <span className="section-number">01</span>
        <div>
          <span className="eyebrow">THE STORY SO FAR</span>
          <h2 id="summary-heading">AI event summary</h2>
        </div>
        <Sparkles size={22} />
      </div>
      {doc ? (
        <>
          <div className="intelligence-summary">
            <span className="eyebrow">WHAT HAPPENED</span>
            <CitedText
              fact={doc.what_happened}
              citations={doc.citations}
              scope="summary"
            />
            <div className="summary-provenance">
              Based on {doc.article_count} of {doc.total_event_articles} reports
              from {doc.source_count} sources · Generated{" "}
              {date(view.generated_at, true)}
            </div>
          </div>
          <div className="key-facts">
            <h3>
              <ListChecks size={19} /> Key facts
            </h3>
            {doc.key_facts.length ? (
              <ol>
                {doc.key_facts.map((fact, i) => (
                  <li key={i}>
                    <CitedText
                      fact={fact}
                      citations={doc.citations}
                      scope="summary"
                    />
                  </li>
                ))}
              </ol>
            ) : (
              <Notice>
                No additional key facts were extracted from the available
                reporting.
              </Notice>
            )}
          </div>
          {doc.latest_update && (
            <div className="latest-report">
              <span className="eyebrow">
                LATEST INCLUDED REPORT · {date(doc.latest_reported_at, true)}
              </span>
              <CitedText
                fact={doc.latest_update}
                citations={doc.citations}
                scope="summary"
              />
            </div>
          )}
          {!!doc.omitted_article_ids.length && (
            <Notice>
              {doc.omitted_article_ids.length} event reports are outside this
              summary. See related articles for the full group.
            </Notice>
          )}
          <Evidence citations={doc.citations} scope="summary" />
        </>
      ) : (
        <>
          <Notice>
            {stale
              ? "This summary is out of date because the underlying coverage changed. It is hidden until the event is refreshed."
              : view.status === "insufficient_sources"
                ? "A cited event summary needs at least two sources. Read the available reports below."
                : "A cited summary is not available for this event yet. Read the available reports below."}
          </Notice>
          <h3 className="empty-subheading">Key facts</h3>
          <p className="muted-copy">
            Key facts will appear when a current cited summary is available.
          </p>
        </>
      )}
    </section>
  );
}

export function ComparisonContent({
  value,
  eventStale = false,
}: {
  value: EventComparison;
  eventStale?: boolean;
}) {
  const stale = eventStale || value.stale || value.status === "stale";
  const doc = !stale && value.status === "compared" ? value.document : null;
  if (!doc)
    return (
      <>
        <h3>What sources agree on</h3>
        <Notice>
          {stale
            ? "Comparison is out of date. Changed reports must be regrouped before comparing them."
            : value.status === "insufficient_sources"
              ? "At least two independent sources are needed for a comparison."
              : value.status === "unsupported_language"
                ? "Comparison is not available for the languages in these reports."
                : "There is not enough usable text to compare these sources."}
        </Notice>
        <h3 className="empty-subheading">Different perspectives</h3>
        <p className="muted-copy">
          No comparison is available. This does not establish agreement or
          disagreement.
        </p>
      </>
    );
  return (
    <>
      <div className="comparison-scope">
        <GitCompareArrows size={17} />
        <span>
          {doc.source_count} sources · {doc.article_count} of{" "}
          {doc.total_event_articles} reports compared
        </span>
      </div>
      <div className="agreement-panel">
        <h3>What sources agree on</h3>
        <p className="intelligence-caption">
          Matching wording in the compared excerpts; not independent
          verification.
        </p>
        {doc.agreement.length ? (
          doc.agreement.map((fact, i) => (
            <CitedText
              key={i}
              fact={fact}
              citations={doc.citations}
              scope="comparison"
            />
          ))
        ) : (
          <Notice>
            No matching wording was identified. This does not mean the sources
            disagree.
          </Notice>
        )}
      </div>
      <h3 className="perspectives-heading">Different perspectives</h3>
      <p className="intelligence-caption">
        Differences can reflect timing, wording or attribution. They are not
        proven contradictions.
      </p>
      {doc.differences.map((difference, i) => (
        <div className="perspective-card" key={i}>
          <span className="perspective-tag">
            {difference.kind === "numbers"
              ? "Numbers in the reporting"
              : "Wording in the reporting"}
          </span>
          <p>{difference.explanation}</p>
          {difference.reports.map((fact, j) => (
            <CitedText
              key={j}
              fact={fact}
              citations={doc.citations}
              scope="comparison"
            />
          ))}
        </div>
      ))}
      {doc.emphasis.map((item, i) => (
        <div className="perspective-card" key={`emphasis-${i}`}>
          <span className="perspective-tag">
            Reporting emphasis · {item.topic}
          </span>
          <p>{item.explanation}</p>
          {item.reports.map((fact, j) => (
            <CitedText
              key={j}
              fact={fact}
              citations={doc.citations}
              scope="comparison"
            />
          ))}
          {!!item.other_compared_sources.length && (
            <p className="intelligence-caption">
              Other compared excerpts: {item.other_compared_sources.join(", ")}.
              Absence here does not establish an omission from the full report.
            </p>
          )}
        </div>
      ))}
      {!doc.differences.length && !doc.emphasis.length && (
        <Notice>
          No distinct differences or emphasis were identified in these excerpts.
        </Notice>
      )}
      {!!doc.additional_details.length && (
        <div className="additional-reporting">
          <h3>Other details in the reporting</h3>
          <p className="intelligence-caption">
            Unmatched wording, which may include paraphrases of the same facts.
          </p>
          {doc.additional_details.map((fact, i) => (
            <CitedText
              key={i}
              fact={fact}
              citations={doc.citations}
              scope="comparison"
            />
          ))}
        </div>
      )}
      {!!doc.omitted_article_ids.length && (
        <Notice>
          {doc.omitted_article_ids.length} event reports were not included in
          this comparison.
        </Notice>
      )}
      <details className="scope-disclosure">
        <summary>How to read this comparison</summary>
        <ul>
          {doc.scope.map((note, i) => (
            <li key={i}>{note}</li>
          ))}
        </ul>
        <p>Computed {date(value.computed_at, true)}</p>
      </details>
      <Evidence citations={doc.citations} scope="comparison" />
    </>
  );
}

function ComparisonSection({ event }: { event: NewsEvent }) {
  const result = useApi<EventComparison>(`/events/${event.id}/comparison`);
  return (
    <section
      id="event-comparison"
      className="intelligence-section"
      aria-labelledby="comparison-heading"
    >
      <div className="intelligence-section-heading">
        <span className="section-number">02</span>
        <div>
          <span className="eyebrow">ACROSS THE SOURCES</span>
          <h2 id="comparison-heading">Compare the coverage</h2>
        </div>
        <GitCompareArrows size={22} />
      </div>
      {result.loading ? (
        <Loading label="Comparing source excerpts" />
      ) : result.error ? (
        <ErrorState message={result.error} retry={result.retry} />
      ) : (
        result.data && (
          <ComparisonContent
            value={result.data}
            eventStale={event.stale || result.data.event_id !== event.id}
          />
        )
      )}
    </section>
  );
}

function ReportMoment({
  item,
  undated = false,
}: {
  item: TimelineItem;
  undated?: boolean;
}) {
  return (
    <li className="report-moment">
      <span className="timeline-dot" />
      <div className="moment-meta">
        <time
          dateTime={!undated && item.timestamp ? item.timestamp : undefined}
        >
          {undated
            ? "Publication date unavailable"
            : date(item.timestamp, true)}
        </time>
        <span>{item.citation.source}</span>
      </div>
      <p>{item.description}</p>
      <Link
        className="timeline-article"
        href={`/articles/${item.source_article_id}`}
      >
        {item.citation.title}
        <ArrowUpRight size={14} />
      </Link>
      <SourceLink url={item.citation.url}>Original report</SourceLink>
      <details>
        <summary>View source excerpt</summary>
        <blockquote>{item.citation.excerpt}</blockquote>
      </details>
    </li>
  );
}

export function TimelineContent({
  value,
  eventStale = false,
  onPage,
}: {
  value: EventTimeline;
  eventStale?: boolean;
  onPage: (offset: number) => void;
}) {
  const stale = eventStale || value.stale || value.status === "stale";
  const doc = !stale && value.status === "ready" ? value.document : null;
  if (!doc)
    return (
      <Notice>
        {stale
          ? "The timeline is out of date. Changed reports must be regrouped before a current timeline can be shown."
          : "No usable reporting is available for this timeline yet."}
      </Notice>
    );
  return (
    <>
      <p className="intelligence-caption">
        Article publication times in UTC, oldest first. These are report dates,
        not inferred times when events occurred.
      </p>
      {!!doc.items.length && (
        <ol className="report-timeline">
          {doc.items.map((item) => (
            <ReportMoment key={item.id} item={item} />
          ))}
        </ol>
      )}
      {!!doc.undated_items.length && (
        <>
          <h3 className="undated-heading">Undated reports</h3>
          <ol className="report-timeline undated">
            {doc.undated_items.map((item) => (
              <ReportMoment key={item.id} item={item} undated />
            ))}
          </ol>
        </>
      )}
      {!doc.count && (
        <Notice>No reports on this page. Return to the previous page.</Notice>
      )}
      <div className="timeline-pagination" aria-label="Timeline pagination">
        <span>
          {doc.count ? `${doc.offset + 1}–${doc.offset + doc.count}` : "0"} of{" "}
          {doc.total_count} reports · {doc.undated_count} undated
        </span>
        <div>
          <button
            className="button secondary small"
            disabled={doc.offset === 0}
            onClick={() => onPage(Math.max(0, doc.offset - doc.limit))}
          >
            Earlier reports
          </button>
          <button
            className="button secondary small"
            disabled={doc.next_offset === null}
            onClick={() => {
              if (doc.next_offset !== null) onPage(doc.next_offset);
            }}
          >
            Later reports
          </button>
        </div>
      </div>
      {!!doc.excluded_articles.length && (
        <details className="scope-disclosure">
          <summary>
            {doc.excluded_articles.length} reports excluded from the timeline
          </summary>
          <ul>
            {doc.excluded_articles.map((item) => (
              <li key={item.article_id}>
                <Link href={`/articles/${item.article_id}`}>
                  Article {item.article_id}
                </Link>
                :{" "}
                {item.reason === "unsupported_language"
                  ? "language not supported"
                  : "not enough usable text"}
              </li>
            ))}
          </ul>
        </details>
      )}
      <details className="scope-disclosure">
        <summary>Timeline coverage and limits</summary>
        <ul>
          {doc.scope.map((note, i) => (
            <li key={i}>{note}</li>
          ))}
        </ul>
        <p>Computed {date(value.computed_at, true)}</p>
      </details>
    </>
  );
}

function TimelineSection({ event }: { event: NewsEvent }) {
  const [offset, setOffset] = useState(0);
  const result = useApi<EventTimeline>(
    `/events/${event.id}/timeline?limit=10&offset=${offset}`,
  );
  return (
    <section
      id="event-timeline"
      className="intelligence-section"
      aria-labelledby="timeline-heading"
    >
      <div className="intelligence-section-heading">
        <span className="section-number">03</span>
        <div>
          <span className="eyebrow">FOLLOW THE REPORTING</span>
          <h2 id="timeline-heading">Timeline</h2>
        </div>
        <Clock3 size={22} />
      </div>
      {result.loading ? (
        <Loading label="Building the reporting timeline" />
      ) : result.error ? (
        <ErrorState message={result.error} retry={result.retry} />
      ) : (
        result.data && (
          <TimelineContent
            value={result.data}
            eventStale={event.stale || result.data.event_id !== event.id}
            onPage={setOffset}
          />
        )
      )}
    </section>
  );
}

export function EventSidebar({ event }: { event: NewsEvent }) {
  const [source, setSource] = useState("");
  const sources = [
    ...new Set(event.articles.map((article) => article.source)),
  ].sort();
  const articles = event.articles.filter(
    (article) => !source || article.source === source,
  );
  return (
    <aside className="event-sidebar">
      <div className="event-ask">
        <Sparkles size={25} />
        <span className="eyebrow">FOLLOW YOUR CURIOSITY</span>
        <h2>There’s more to the story.</h2>
        <p>
          Ask a question inspired by this coverage, then follow the evidence.
        </p>
        <Link
          className="button"
          href={`/chat?q=${encodeURIComponent(`What do the collected reports say about "${event.title}"?`.slice(0, 2000))}`}
        >
          Ask about this story
          <ArrowUpRight size={15} />
        </Link>
        <small>
          Opens a question using this event title. Chat searches the collected
          library.
        </small>
      </div>
      <section
        id="event-entities"
        className="event-side-panel"
        aria-labelledby="entities-heading"
      >
        <div className="side-panel-heading">
          <Users size={18} />
          <h2 id="entities-heading">Related entities</h2>
        </div>
        <p className="intelligence-caption">
          People, organizations and places extracted from the reports.
        </p>
        {!event.stale && event.related_entities?.length ? (
          <ul className="event-entity-list">
            {event.related_entities.map((entity) => (
              <li key={entity.id}>
                <Link href={`/feed?entity_id=${entity.id}`}>
                  <span>{entity.name}</span>
                  <ArrowUpRight size={14} />
                </Link>
                <small>
                  {entity.entity_type.toLowerCase()} ·{" "}
                  {entity.article_ids.length}{" "}
                  {entity.article_ids.length === 1 ? "report" : "reports"}
                </small>
              </li>
            ))}
          </ul>
        ) : (
          <Notice>
            {event.stale
              ? "Entity associations are hidden until this event is refreshed."
              : "No related entities have been extracted from the current reports."}
          </Notice>
        )}
      </section>
      <section
        id="event-articles"
        className="event-side-panel"
        aria-labelledby="articles-heading"
      >
        <div className="side-panel-heading">
          <BookOpen size={18} />
          <h2 id="articles-heading">Related articles</h2>
          <span>{event.article_count}</span>
        </div>
        <label className="event-source-filter">
          Publisher
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="">All sources</option>
            {sources.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </label>
        <p className="intelligence-caption">
          Showing {articles.length} of {event.articles.length} reports
        </p>
        <div className="event-related-list">
          {articles.map((article) => (
            <article key={article.article_id}>
              <div className="eyebrow">{article.source}</div>
              <h3>
                <Link href={`/articles/${article.article_id}`}>
                  {article.title}
                </Link>
              </h3>
              <time dateTime={article.published_at || undefined}>
                {date(article.published_at, true)}
              </time>
              {article.stale && (
                <p className="member-stale">Changed since grouping</p>
              )}
              <SourceLink url={article.url}>Original report</SourceLink>
            </article>
          ))}
        </div>
        {!articles.length && <Notice>No related reports are available.</Notice>}
      </section>
    </aside>
  );
}

export function EventDetail({ id }: { id: string }) {
  const result = useApi<NewsEvent>(`/events/${id}`);
  if (result.loading) return <Loading label="Opening event intelligence" />;
  if (result.error)
    return (
      <>
        <Link className="back-link" href="/events">
          <ArrowLeft size={15} />
          All news events
        </Link>
        <ErrorState message={result.error} retry={result.retry} />
      </>
    );
  const event = result.data!;
  return (
    <div className="event-intelligence">
      <Link className="back-link" href="/events">
        <ArrowLeft size={15} />
        All news events
      </Link>
      <header className="intelligence-hero">
        <div className="intelligence-kicker">
          <span className="eyebrow">
            <Layers3 size={15} /> EVENT INTELLIGENCE
          </span>
          <span className="event-edition">THE CONNECTED VIEW</span>
        </div>
        <h1>{event.title}</h1>
        <p>
          One story, viewed through the reporting. Trace the facts, compare the
          coverage, and see how it unfolded.
        </p>
        <div className="intelligence-stats">
          <span>
            <BookOpen size={17} />
            <strong>{event.source_count}</strong>{" "}
            {event.source_count === 1 ? "source" : "sources"}
          </span>
          <span>
            <Layers3 size={17} />
            <strong>{event.article_count}</strong> reports
          </span>
          <span>
            <Clock3 size={17} />
            Updated{" "}
            <time dateTime={event.last_updated_at}>
              {date(event.last_updated_at, true)}
            </time>
          </span>
        </div>
      </header>
      {event.stale && (
        <div role="status" className="event-stale-banner">
          Coverage has changed since this event was grouped. Summary,
          comparison, timeline and entity insights are hidden until refreshed.
          Original article links remain available.
        </div>
      )}
      <nav className="event-jump-nav" aria-label="Event sections">
        <a href="#event-summary">The summary</a>
        <a href="#event-comparison">Source perspectives</a>
        <a href="#event-timeline">Timeline</a>
        <a href="#event-entities">Entities</a>
        <a href="#event-articles">
          Articles <ExternalLink size={12} />
        </a>
      </nav>
      <div className="intelligence-layout">
        <div className="intelligence-main">
          <SummarySection event={event} />
          <ComparisonSection event={event} />
          <TimelineSection event={event} />
        </div>
        <EventSidebar event={event} />
      </div>
      <p className="event-footnote">
        Built from collected reporting. Source agreement is not proof,
        publication time is not event time, and automated extraction can miss
        context. Follow the original sources.
      </p>
    </div>
  );
}
