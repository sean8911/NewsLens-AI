"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  ArrowUpRight,
  ChartNoAxesCombined,
  Layers3,
  Search,
  Sparkles,
} from "lucide-react";
import { date, useApi } from "@/lib/api";
import type { Article, EventPage, Page, Topic, Trends } from "@/lib/types";
import {
  ArticleCard,
  AskBanner,
  Empty,
  ErrorState,
  EventCard,
  Loading,
} from "./ui";
import { categories } from "./shell";

export function Home() {
  const router = useRouter();
  const articles = useApi<Page<Article>>("/articles?limit=7");
  const topics = useApi<Page<Topic>>("/topics?limit=5");
  const trends = useApi<Trends>("/trending?limit=3");
  const events = useApi<EventPage>("/events?limit=2");
  const lead = articles.data?.results[0];
  return (
    <>
      <section className="home-intro">
        <div>
          <div className="eyebrow">
            <span className="live-dash" />
            THE BIGGER PICTURE
          </div>
          <h1>
            The world moves fast.
            <br />
            <em>Make sense of it.</em>
          </h1>
          <p>
            Your news, connected. Explore the stories, sources, and ideas
            shaping our world.
          </p>
        </div>
        <div className="edition-note">
          <span>THE NEWSLENS BRIEF</span>
          <strong>
            A little context.
            <br />A lot more clarity.
          </strong>
          <div className="edition-rule" />
        </div>
      </section>
      <form
        className="hero-search"
        onSubmit={(event) => {
          event.preventDefault();
          const q = String(
            new FormData(event.currentTarget).get("q") || "",
          ).trim();
          if (q) router.push(`/search?q=${encodeURIComponent(q)}`);
        }}
      >
        <Search size={21} />
        <label className="sr-only" htmlFor="home-query">
          Search your news
        </label>
        <input
          id="home-query"
          name="q"
          placeholder="What would you like to understand?"
          required
          maxLength={2000}
        />
        <button className="button dark" type="submit">
          Explore
          <ArrowRight size={16} />
        </button>
      </form>
      <div className="home-topics">
        <span>Explore</span>
        {categories.map((category) => (
          <Link key={category} href={`/category/${category}?kind=topic`}>
            {category}
            <ArrowUpRight size={12} />
          </Link>
        ))}
      </div>
      <div className="section-heading">
        <div>
          <span className="eyebrow">FROM YOUR LIBRARY</span>
          <h2>In focus</h2>
        </div>
        <Link href="/feed" className="text-link">
          View all stories
          <ArrowRight size={15} />
        </Link>
      </div>
      {articles.error ? (
        <ErrorState message={articles.error} retry={articles.retry} />
      ) : articles.loading ? (
        <Loading />
      ) : !lead ? (
        <Empty title="Your next perspective starts here.">
          Your library is empty. Collected stories will appear here as they
          arrive.
        </Empty>
      ) : (
        <div className="focus-grid">
          <ArticleCard article={lead} feature />
          <div className="context-panel">
            <span className="panel-overline">
              <Layers3 size={16} />A DIFFERENT WAY TO READ
            </span>
            <h2>
              Less noise.
              <br />
              More understanding.
            </h2>
            <p>
              Find connections across coverage. Go from a headline to the
              evidence behind it.
            </p>
            <div className="source-orbit" aria-hidden="true">
              <div className="orbit-line" />
              <span className="orbit-source source-one">Reporting</span>
              <span className="orbit-core">
                <Sparkles size={27} />
              </span>
              <span className="orbit-source source-two">Context</span>
              <span className="orbit-source source-three">Perspective</span>
            </div>
            <Link href="/events" className="panel-link">
              Explore connected stories
              <ArrowUpRight size={17} />
            </Link>
          </div>
        </div>
      )}
      <div className="home-lower-grid">
        <section>
          <div className="section-heading">
            <div>
              <span className="eyebrow">KEEP EXPLORING</span>
              <h2>The latest in your library</h2>
            </div>
            <span className="subtle-count">
              {articles.data?.total ?? "—"} articles
            </span>
          </div>
          {lead && (
            <div className="article-grid two-col">
              {articles.data?.results.slice(1, 5).map((article) => (
                <ArticleCard key={article.id} article={article} />
              ))}
            </div>
          )}
          {events.data && events.data.count > 0 && (
            <>
              <div className="section-heading">
                <h2>Stories coming together</h2>
                <Link className="text-link" href="/events">
                  All events
                  <ArrowRight size={15} />
                </Link>
              </div>
              <div className="article-grid two-col">
                {events.data.results.map((event) => (
                  <EventCard key={event.id} event={event} />
                ))}
              </div>
            </>
          )}
        </section>
        <aside className="insights-column">
          <div className="section-heading">
            <h2>
              <ChartNoAxesCombined size={19} />
              On the radar
            </h2>
          </div>
          <div className="radar-box">
            <span className="eyebrow">LAST 24 HOURS</span>
            {trends.error ? (
              <p className="muted-copy">
                Trending coverage is temporarily unavailable.
              </p>
            ) : trends.loading ? (
              <p className="muted-copy">Looking across the sources…</p>
            ) : trends.data?.count ? (
              trends.data.results.map((trend, index) => (
                <Link
                  className="trend-row"
                  href={`/category/${encodeURIComponent(trend.topic)}?kind=topic`}
                  key={trend.id}
                >
                  <span className="trend-rank">0{index + 1}</span>
                  <div>
                    <strong>{trend.label}</strong>
                    <small>
                      {trend.source_count} sources · {trend.article_count}{" "}
                      articles
                    </small>
                  </div>
                  <ArrowUpRight size={15} />
                </Link>
              ))
            ) : (
              <div className="quiet-trends">
                <ChartNoAxesCombined size={27} />
                <h3>A quieter news window.</h3>
                <p>
                  No multi-source trends in the last 24 hours. Explore a wider
                  view.
                </p>
              </div>
            )}
            <Link className="text-link" href="/trending">
              Explore trends
              <ArrowRight size={15} />
            </Link>
          </div>
          <div className="topics-box">
            <span className="eyebrow">IN THE COLLECTION</span>
            <h3>Follow your curiosity.</h3>
            {topics.data?.results.map((topic) => (
              <Link
                key={topic.topic}
                href={`/category/${encodeURIComponent(topic.topic)}?kind=topic`}
              >
                <span>{topic.topic}</span>
                <small>{topic.article_count}</small>
              </Link>
            ))}
            {!topics.data?.count && (
              <p className="muted-copy">Browse a category to discover more.</p>
            )}
          </div>
          {lead && (
            <div className="collection-note">
              Latest publication in this collection
              <br />
              <strong>{date(lead.published_at)}</strong>
              <p>
                Publication dates stay visible, so older coverage never passes
                for breaking news.
              </p>
            </div>
          )}
        </aside>
      </div>
      <AskBanner />
    </>
  );
}
