from backend.app.conversation_runtime import conversation_store
from backend.app.core.config import Settings
from backend.app.core.performance import measured, monitored
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.search_runtime import search_snapshot
from backend.app.services.chat import answer_question
from backend.app.services.chat_context import collect_evidence
from backend.app.services.conversations import run_conversation
from nlp.llm.local import language_model_for
from nlp.multilingual.local_translation import LocalTranslator


@monitored("chat")
@measured("chatbot")
def run_chat(request: ChatRequest, settings: Settings | None = None) -> ChatResponse:
    settings = settings or Settings()
    if request.conversation_id is not None:
        with conversation_store(settings) as store:
            return run_conversation(
                request,
                store,
                lambda effective, resolved, context: _run_single(
                    effective, settings, resolved, context
                ),
            )
    return _run_single(request, settings)


def _run_single(request, settings, resolved_question=None, conversation_context=None):
    search_request = request.search_request()
    if resolved_question is not None:
        search_request = search_request.model_copy(update={"q": resolved_question})
    time_options = {"time_query": request.question} if resolved_question is not None else {}
    with search_snapshot(search_request, settings, **time_options) as (retrieval, sessions):
        evidence = collect_evidence(
            retrieval,
            search_request,
            sessions,
            settings.multilingual_chat_min_reranking_score
            if settings.multilingual_enabled
            else settings.chat_min_reranking_score,
        )
    # Release SQL transaction and Qdrant handles before potentially slow generation.
    # Long history plus evidence can exhaust the CPU generation deadline during prefill.
    # Bound every persistent turn, including the initial topic, through whole-excerpt fitting.
    model_settings = settings
    if resolved_question is not None:
        model_settings = settings.model_copy(
            update={
                "chat_max_input_tokens": min(
                    settings.chat_max_input_tokens, settings.chat_conversation_max_input_tokens
                ),
                "chat_generation_seconds": settings.chat_conversation_generation_seconds,
            }
        )
    response = answer_question(
        " ".join(request.question.split()),
        evidence,
        lambda: language_model_for(model_settings),
        retrieval.reranking.model_version if retrieval.reranking else None,
        as_of=retrieval.temporal.intent.as_of if retrieval.temporal else None,
        answer_language=request.answer_language,
        translator=LocalTranslator(settings),
        contextual_question=resolved_question if conversation_context else None,
        conversation_context=conversation_context,
    )

    return response.model_copy(
        update={"time_intent": retrieval.temporal.intent if retrieval.temporal else None}
    )
