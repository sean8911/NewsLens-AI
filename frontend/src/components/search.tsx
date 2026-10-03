"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Search as SearchIcon,
  Sparkles,
} from "lucide-react";
import { api, date } from "@/lib/api";
import type { SearchResult } from "@/lib/types";
import { Empty, ErrorState, Header, Loading, SourceLink } from "./ui";

export function SearchPage() {
  const params = useSearchParams();
  const router = useRouter();
  const q = (params.get("q") || "").slice(0, 2000);
  const selected = params.get("method") || "semantic";
  const method = ["semantic", "hybrid", "lexical"].includes(selected)
    ? selected
    : "semantic";
  const rerank = params.get("rerank") === "true";
  const signature = JSON.stringify({ q, method, rerank });
  const [revision, setRevision] = useState(0);
  const key = signature + revision;
  const [result, setResult] = useState<{
    key: string;
    data?: SearchResult;
    error?: string;
  }>({ key: "" });
  useEffect(() => {
    if (!q.trim()) return;
    const controller = new AbortController();
    api<SearchResult>("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q, method, rerank, limit: 12 }),
      signal: controller.signal,
    }).then(
      (data) => {
        if (!controller.signal.aborted) setResult({ key, data });
      },
      (error) => {
        if (!controller.signal.aborted)
          setResult({ key, error: error.message });
      },
    );
    return () => controller.abort();
  }, [q, method, rerank, key]);
  const current = result.key === key ? result : undefined;
  return (
    <>
      <Header
        eyebrow="FOLLOW AN IDEA"
        title="Find meaning, not just keywords."
        description="Explore your news by concept, question, or exact phrase."
      />
      <form
        className="search-workbench"
        key={signature}
        onSubmit={(event) => {
          event.preventDefault();
          const data = new FormData(event.currentTarget);
          const next = new URLSearchParams();
          next.set("q", String(data.get("q") || "").trim());
          next.set("method", String(data.get("method")));
          if (data.get("rerank")) next.set("rerank", "true");
          if (
            next.get("q") === q &&
            next.get("method") === method &&
            (next.get("rerank") === "true") === rerank
          ) {
            setRevision((value) => value + 1);
          }
          router.push(`/search?${next}`);
        }}
      >
        <div className="search-input-row">
          <SearchIcon size={22} />
          <label className="sr-only" htmlFor="search-query">
            Search question
          </label>
          <input
            id="search-query"
            name="q"
            placeholder="e.g. How is AI changing the way we work?"
            defaultValue={q}
            required
            maxLength={2000}
            pattern=".*\S.*"
          />
          <button className="button dark" type="submit">
            Search
            <ArrowRight size={16} />
          </button>
        </div>
        <div className="search-options">
          <label>
            Search by
            <select name="method" defaultValue={method}>
              <option value="semantic">Meaning</option>
              <option value="hybrid">Meaning + keywords</option>
              <option value="lexical">Keywords</option>
            </select>
          </label>
          <label className="checkbox-label">
            <input type="checkbox" name="rerank" defaultChecked={rerank} />
            Refine the ranking
          </label>
          <span>Refinement takes a little longer.</span>
        </div>
      </form>
      {!q.trim() ? (
        <div className="search-start">
          <span className="eyebrow">A FEW PLACES TO START</span>
          <div className="suggestion-grid">
            {[
              "What is changing in artificial intelligence?",
              "What is the Xbox console limit?",
              "How are businesses responding to climate change?",
            ].map((question) => (
              <Link
                key={question}
                href={`/search?q=${encodeURIComponent(question)}`}
              >
                <Sparkles size={18} />
                <span>{question}</span>
                <ArrowUpRight size={16} />
              </Link>
            ))}
          </div>
          <p className="data-note">
            Search explores your collected articles. It does not browse the live
            web.
          </p>
        </div>
      ) : !current ? (
        <Loading
          label={
            rerank
              ? "Finding and refining the most relevant reporting"
              : "Finding the reporting behind your question"
          }
        />
      ) : current.error ? (
        <ErrorState
          message={current.error}
          retry={() => setRevision((value) => value + 1)}
        />
      ) : (
        current.data && (
          <>
            <div className="results-topline">
              <span>
                {current.data.count} ranked{" "}
                {current.data.count === 1 ? "result" : "results"}
              </span>
              <span>
                {method === "lexical"
                  ? "Keyword"
                  : method === "hybrid"
                    ? "Hybrid"
                    : "Semantic"}{" "}
                search{rerank ? " · Refined ranking" : ""}
              </span>
            </div>
            {current.data.count ? (
              <div className="search-results">
                {current.data.results.map((hit, i) => (
                  <article className="search-hit" key={hit.article_id}>
                    <span className="hit-number">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <div>
                      <div className="eyebrow">
                        {hit.category} · {hit.source}
                      </div>
                      <h2>
                        <Link href={`/articles/${hit.article_id}`}>
                          {hit.title}
                        </Link>
                      </h2>
                      <p>
                        {hit.summary ||
                          "Read the collected article for the details and original source."}
                      </p>
                      <div className="hit-footer">
                        <span>{date(hit.published_at)}</span>
                        <SourceLink url={hit.url}>Original report</SourceLink>
                      </div>
                    </div>
                    <Link
                      className="round-arrow"
                      href={`/articles/${hit.article_id}`}
                      aria-label={`Read ${hit.title}`}
                    >
                      <ArrowUpRight size={18} />
                    </Link>
                  </article>
                ))}
              </div>
            ) : (
              <Empty title="No reporting matched this search.">
                Try a broader question or switch between meaning and keywords.
              </Empty>
            )}
          </>
        )
      )}
    </>
  );
}
