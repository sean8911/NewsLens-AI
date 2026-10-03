import logging
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from backend.app.schemas.chat import ChatResponse, ChatSource, EvidenceSelection
from backend.app.services.chat_context import POLICY, Evidence, messages_for
from nlp.embeddings.service import digest
from nlp.multilingual.grounding import required_fact_present
from nlp.multilingual.language import resolve_languages, retrieval_question
from nlp.multilingual.translation import INSUFFICIENT as LOCALIZED_INSUFFICIENT
from nlp.multilingual.translation import INTRO, TRANSLATED, translate_excerpt
from nlp.multilingual.translation import POLICY as TRANSLATION_POLICY

logger = logging.getLogger(__name__)
INSUFFICIENT = "I don't have enough information in the available articles to answer that question."


class LanguageModel(Protocol):
    version: str

    def generate(self, messages: list[dict]) -> str: ...


def answer_question(
    question: str,
    evidence: list[Evidence],
    provider_factory: Callable[[], LanguageModel],
    reranking_version: str | None = None,
    *,
    as_of: datetime | None = None,
    answer_language: str | None = None,
    translator=None,
    contextual_question: str | None = None,
    conversation_context: dict | None = None,
) -> ChatResponse:
    detected, output_language = resolve_languages(question, answer_language)
    context_hash = digest(
        [POLICY, TRANSLATION_POLICY, output_language, [item.prompt_value() for item in evidence]]
    )
    if contextual_question is not None:
        context_hash = digest(
            [
                POLICY,
                TRANSLATION_POLICY,
                output_language,
                contextual_question,
                conversation_context,
                [item.prompt_value() for item in evidence],
            ]
        )
    result = ChatResponse(
        question=question,
        answer=LOCALIZED_INSUFFICIENT[output_language],
        query_language=detected.code,
        answer_language=output_language,
        language_detection_fallback=detected.fallback,
        status="insufficient_information",
        reason="no_evidence",
        context_hash=context_hash,
        reranking_model_version=reranking_version,
    )
    if not evidence:
        return result
    provider = provider_factory()
    result.llm_version = provider.version
    evidence = list(evidence)
    prompt_question = retrieval_question(question)
    messages = messages_for(
        prompt_question, evidence, as_of, conversation_context, contextual_question
    )
    fits = getattr(provider, "context_fits", None)
    if fits is not None:
        if conversation_context and not fits(messages):
            conversation_context = None
            messages = messages_for(
                prompt_question, evidence, as_of, resolved_question=contextual_question
            )
        while evidence and not fits(messages):
            # Keep whole excerpts and prefer the highest-ranked sources.
            evidence.pop()
            messages = messages_for(
                prompt_question, evidence, as_of, conversation_context, contextual_question
            )
        if not evidence:
            raise ValueError("No evidence fits the chat token budget")
        result.context_hash = digest(
            [
                POLICY,
                TRANSLATION_POLICY,
                output_language,
                [item.prompt_value() for item in evidence],
            ]
        )
    if contextual_question is not None:
        result.context_hash = digest(
            [
                POLICY,
                TRANSLATION_POLICY,
                output_language,
                prompt_question,
                contextual_question,
                conversation_context,
                [item.prompt_value() for item in evidence],
            ]
        )
        result.history_turns_used = len((conversation_context or {}).get("recent_turns", []))
        result.memory_used = bool(conversation_context)
    raw = provider.generate(messages)
    try:
        if len(raw) > 4096:
            raise ValueError("Oversized model output")
        selected = EvidenceSelection.model_validate_json(raw).evidence_ids
        known = {item.id: item for item in evidence}
        if len(set(selected)) != len(selected) or any(id_ not in known for id_ in selected):
            raise ValueError("Invalid evidence references")
    except ValueError:
        logger.warning("chat_invalid_model_output")
        return result.model_copy(update={"reason": "invalid_model_output"})
    if not selected:
        return result.model_copy(update={"reason": "model_abstained"})
    if not required_fact_present(question, [known[id_].text for id_ in selected]):
        return result.model_copy(update={"reason": "unsupported_fact"})
    # Remove a headline only when its complete text appears in a selected body excerpt.
    # Headlines can contain unique facts (e.g. why an event matters); keep those facts.
    selected = [
        id_
        for id_ in selected
        if known[id_].text != known[id_].title
        or not any(
            known[other].article_id == known[id_].article_id
            and known[other].text != known[other].title
            and known[id_].text.casefold() in known[other].text.casefold()
            for other in selected
        )
    ]
    sources: dict[int, ChatSource] = {}
    lines = []
    for id_ in selected:
        item = known[id_]
        try:
            rendered, translated = translate_excerpt(item.text, output_language, translator)
        except ValueError:
            logger.warning("chat_invalid_translation")
            return result.model_copy(update={"reason": "invalid_model_output"})
        if translated:
            result.translation_model_version = translator.version
        if item.article_id not in sources:
            sources[item.article_id] = ChatSource(
                citation_id=len(sources) + 1,
                article_id=item.article_id,
                title=item.title,
                source=item.source,
                published_at=item.published_at,
                retrieved_at=item.retrieved_at,
                url=item.url,
                excerpts=[],
            )
        citation = sources[item.article_id]
        citation.excerpts.append(item.text)
        citation.translations.append(rendered if translated else None)
        label = TRANSLATED[output_language] + ": " if translated else ""
        lines.append(f"{label}“{rendered}” [{citation.citation_id}]")
    return result.model_copy(
        update={
            "status": "answered",
            "reason": None,
            "answer": INTRO[output_language] + "\n\n" + "\n\n".join(lines),
            "sources": list(sources.values()),
        }
    )
