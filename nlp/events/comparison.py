"""Bounded, attributed observations about reporting; no bias or truth inference."""

import re
from difflib import SequenceMatcher
from itertools import combinations

from segtok.segmenter import split_single

from backend.app.schemas.event_comparison import (
    EventComparisonDocument,
    ReportDifference,
    ReportingEmphasis,
)
from backend.app.schemas.event_summary import SummaryFact
from backend.app.services.event_summaries import validate_grounding
from nlp.embeddings.service import digest
from nlp.events.summarization import CAVEAT, POLICY_VERSION, EventSummarizer, publisher_key

TOPICS = {
    "Financial terms": (
        r"\b(?:profit|profits|revenue|shares|shareholders|billion|million|cost|costs|funding)\b"
    ),
    "Regulatory terms": (
        r"\b(?:regulator|regulators|regulatory|antitrust|approval|authorities|court)\b"
    ),
    "Employee terms": r"\b(?:employees|workers|jobs|layoffs|hiring|hire|union|staff)\b",
    "Technical terms": (
        r"\b(?:processor|chip|semiconductor|software|hardware|performance|technology)\b"
    ),
}
VERSION = digest(["observable-report-comparison-v1", POLICY_VERSION, TOPICS, 0.75])
SCOPE = [
    "Comparison covers lead excerpts from up to six publishers, one article per publisher.",
    "Agreement means matching wording, not independent confirmation or consensus of all sources.",
    "No matching wording does not mean that sources disagree.",
    "Additional details have unmatched wording; paraphrases and facts elsewhere may be missed.",
    "Differences may reflect timing, attribution or wording; they are not proven contradictions.",
    "Emphasis records terms in excerpts, not intent, bias, or the entire article's focus.",
]


def passages(excerpt):
    # Keep an explicit qualification with the preceding sentence. The citation also
    # retains the entire selected lead as context for pronouns and attribution.
    result = []
    for piece in split_single(excerpt):
        sentence = piece.strip()
        if result and CAVEAT.match(sentence):
            result[-1] += " " + sentence
        else:
            result.append(sentence)
    return result


def difference_kind(left, right):
    if left == right:
        return None
    # Match numbers only when every other character agrees. Preserve currencies,
    # units, negation and qualifiers so unlike quantities are never reconciled.
    numeric = r"\d+(?:[,.]\d+)*"
    if (
        re.search(numeric, left)
        and re.search(numeric, right)
        and re.sub(numeric, "#", left) == re.sub(numeric, "#", right)
    ):
        return "numbers"
    a, b = re.findall(r"\w+|[^\w\s]", left), re.findall(r"\w+|[^\w\s]", right)
    if min(len(a), len(b)) >= 6 and SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.75:
        return "wording"
    return None


class EventComparator:
    version = VERSION

    def compare(self, sources):
        status, summary = EventSummarizer().summarize(sources)
        if summary is None:
            return status, None
        validate_grounding(summary, sources)
        citations = summary.citations
        by_id = {cite.citation_id: cite for cite in citations}
        facts = {}
        for cite in citations:
            for passage in passages(cite.excerpt):
                fact = facts.setdefault(passage, SummaryFact(text=passage, citation_ids=[]))
                if cite.citation_id not in fact.citation_ids:
                    fact.citation_ids.append(cite.citation_id)

        def publishers(fact):
            return {publisher_key(by_id[id_].source) for id_ in fact.citation_ids}

        agreement = [fact for fact in facts.values() if len(publishers(fact)) >= 2]
        differences, paired = [], set()
        for left, right in combinations(facts.values(), 2):
            # Require an actual pair of different publishers, not just two clauses
            # inside the same article. Keep both quotes instead of guessing truth.
            if len(publishers(left) | publishers(right)) < 2:
                continue
            kind = difference_kind(left.text, right.text)
            if kind:
                differences.append(
                    ReportDifference(
                        kind=kind,
                        explanation=(
                            "Numbers differ in matching wording; verify dates and context."
                            if kind == "numbers"
                            else "Similar passages differ in wording; quotes retain qualifiers."
                        ),
                        reports=[left, right],
                    )
                )
                paired.update([left.text, right.text])
        additional = [
            fact
            for fact in facts.values()
            if len(publishers(fact)) == 1 and fact.text not in paired
        ]
        emphasis = []
        for topic, pattern in TOPICS.items():
            reports = [fact for fact in facts.values() if re.search(pattern, fact.text, re.I)]
            represented = {key for fact in reports for key in publishers(fact)}
            others = [
                cite.source for cite in citations if publisher_key(cite.source) not in represented
            ]
            if reports and others:
                emphasis.append(
                    ReportingEmphasis(
                        topic=topic,
                        explanation=(
                            "These topic terms appear in the cited excerpts; "
                            "no match in the other selected excerpts."
                        ),
                        reports=reports,
                        other_compared_sources=others,
                    )
                )
        document = EventComparisonDocument(
            agreement=agreement,
            differences=differences,
            additional_details=additional,
            emphasis=emphasis,
            citations=citations,
            source_count=summary.source_count,
            article_count=summary.article_count,
            total_event_articles=summary.total_event_articles,
            omitted_article_ids=summary.omitted_article_ids,
            scope=SCOPE,
        )
        return "compared", document
