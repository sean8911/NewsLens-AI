from backend.app.core.config import Settings
from backend.app.core.performance import monitored
from backend.app.schemas.research import ResearchRequest, ResearchResponse
from backend.app.services.research import investigate
from backend.app.services.research_tools import ResearchTools
from nlp.llm.local import language_model_for
from nlp.multilingual.local_translation import LocalTranslator


@monitored("research")
def run_research(request: ResearchRequest, settings: Settings | None = None) -> ResearchResponse:
    settings = settings or Settings()
    model_settings = settings.model_copy(
        update={
            "chat_max_input_tokens": min(
                settings.chat_max_input_tokens, settings.research_input_tokens
            ),
            "chat_generation_seconds": settings.research_generation_seconds,
        }
    )
    return investigate(
        request,
        lambda scope, as_of, intent: ResearchTools(settings, scope, as_of, intent),
        lambda: language_model_for(model_settings),
        translator=LocalTranslator(settings),
        seconds=settings.research_seconds,
    )
