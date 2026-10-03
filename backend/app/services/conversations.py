"""Bounded, extractive topic memory, used for navigation, never as news evidence."""

import json
import re
from datetime import UTC, datetime

from pydantic import ValidationError

from backend.app.schemas.chat import ChatRequest
from backend.app.services.conversation_store import ConversationFiltersInvalid
from backend.app.services.time_intent import resolve_time_intent
from nlp.embeddings.service import digest
from nlp.multilingual.language import resolve_languages

FILTERS = {"source", "language", "category", "date_from", "date_to", "recency"}
CONTEXT_CHARS = 2400
FOLLOW_UP = re.compile(
    r"\b(?:why (?:is|was|does|did) (?:this|that|it)|what (?:happens? next|about)|"
    r"tell me more|explain (?:this|that)|(?:who|where|when) (?:is|was|did) (?:he|she|it)|"
    r"how (?:does|will|did) (?:this|that|it)|(?:mengapa|kenapa).*\b(?:ini|itu)|"
    r"how (?:long|much|many).*\b(?:it|they|that|this)|"
    r"what (?:did|does|will) (?:it|he|she|they)|"
    r"bagaimana pula)\b|(?:为什么|為什麼).{0,8}(?:这|這|那|它|他|她)|"
    r"这.{0,8}(?:重要|意味)|那.{0,8}呢|接下来",
    re.IGNORECASE,
)
CLARIFY = {
    "en": "Which news topic or event do you mean? Please name it in your question.",
    "ms": "Topik berita atau peristiwa yang mana? Sila nyatakannya dalam soalan anda.",
    "zh": "你指的是哪个新闻主题或事件？请在问题中说明。",
}


def prepare(request, memory, turns):
    follow = not request.reset_context and (
        request.follow_up
        if request.follow_up is not None
        else bool(FOLLOW_UP.search(request.question))
    )
    if not follow:
        return request, request.question, None, False
    focus = memory.get("focus_question")
    if not focus:
        return request, None, None, True
    values = request.model_dump()
    inherit = FILTERS - request.model_fields_set
    current_time = resolve_time_intent(request.search_request(), datetime.now(UTC))
    if (
        current_time.expression
        or {"date_from", "date_to"} & request.model_fields_set
        or request.recency == "off"
    ):
        # A new time expression or explicit date bound replaces the previous window.
        inherit -= {"date_from", "date_to", "recency"}
    if turns:
        values.update({k: v for k, v in turns[-1].filters.items() if k in inherit})
    try:
        effective = ChatRequest.model_validate(values)
    except ValidationError:
        raise ConversationFiltersInvalid() from None
    recent = [turn for turn in turns if turn.sequence >= memory.get("topic_start_sequence", 1)]
    context = {
        "topic": focus,
        "source_titles": memory.get("source_titles", []),
        "recent_turns": [
            {"question": t.question[:240], "answer_excerpt": t.response["answer"][:240]}
            for t in recent
        ],
    }
    while len(json.dumps(context, ensure_ascii=False)) > CONTEXT_CHARS:
        if context["recent_turns"]:
            context["recent_turns"].pop(0)
        else:
            context["source_titles"] = []
            break
    anchor = "Topic: " + focus
    if recent and recent[-1].question != focus:
        anchor += "\nPrevious question: " + recent[-1].question[:240]
    if memory.get("source_titles"):
        anchor += "\nArticle titles: " + "; ".join(memory["source_titles"])
    # Never truncate the user's current question, or lose the topic silently.
    resolved = request.question + "\n" + anchor
    if len(resolved) > 2000:
        return effective, None, None, True
    return effective, resolved, context, True


def run_conversation(request, store, answerer):
    memory, turns = store.snapshot(request.conversation_id, request.expected_revision)
    effective, resolved, context, follow = prepare(request, memory, turns)
    if resolved is None:
        from backend.app.schemas.chat import ChatResponse

        detected, target = resolve_languages(request.question, request.answer_language)
        response = ChatResponse(
            question=request.question,
            answer=CLARIFY[target],
            query_language=detected.code,
            answer_language=target,
            language_detection_fallback=detected.fallback,
            status="needs_clarification",
            reason="missing_conversation_context",
            context_hash=digest(["conversation-clarification-v1", request.question]),
        )
    else:
        response = answerer(effective, resolved, context)
    response = response.model_copy(
        update={
            "conversation_id": request.conversation_id,
            "conversation_revision": request.expected_revision + 1,
            "resolved_question": resolved or request.question,
            "follow_up": follow,
        }
    )
    filters = effective.model_dump(mode="json", include=FILTERS)
    if response.time_intent:
        intent = response.time_intent
        if intent.date_from or intent.date_to:
            filters.update(
                date_from=intent.date_from.isoformat() if intent.date_from else None,
                date_to=intent.date_to.isoformat() if intent.date_to else None,
            )
        elif intent.kind == "latest":
            filters["recency"] = "latest"
    if resolved is not None:
        if not follow:
            focus = (
                response.time_intent.retrieval_query if response.time_intent else request.question
            )
            memory = {
                "focus_question": focus[:600],
                "topic_start_sequence": request.expected_revision + 1,
            }
        memory = {**memory, "source_titles": [s.title[:160] for s in response.sources[:3]]}
    store.append(request, response, memory, filters)
    return response
