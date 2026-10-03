from unittest.mock import Mock

import pytest

from backend.app.core.config import Settings
from nlp.llm.local import LocalLanguageModel


@pytest.mark.parametrize("length, fits", [(1024, True), (1025, False), (6000, False)])
def test_token_budget_counts_input_ids_not_tokenizer_dictionary_keys(monkeypatch, length, fits):
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = {
        "input_ids": list(range(length)),
        "attention_mask": [1] * length,
    }
    monkeypatch.setattr("nlp.llm.local._tokenizer", lambda path: tokenizer)
    provider = LocalLanguageModel(Settings(chat_max_input_tokens=1024))
    messages = [{"role": "user", "content": "中文新闻 " * 100}]
    assert provider.context_fits(messages) is fits
    tokenizer.apply_chat_template.assert_called_once_with(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True
    )


@pytest.mark.parametrize("repeats_headline", [True, False])
def test_selected_headline_keeps_unique_facts(repeats_headline):
    from backend.app.services.chat import answer_question
    from backend.app.services.chat_context import Evidence

    title = "The court ruling sets a national precedent."
    body = "The decision was announced on Tuesday."
    if repeats_headline:
        body = title + " " + body
    evidence = [
        Evidence("E1", 1, title, "Example", None, "https://example.com/a", title),
        Evidence("E2", 1, title, "Example", None, "https://example.com/a", body),
    ]
    provider = Mock(version="test", generate=Mock(return_value='{"evidence_ids":["E1","E2"]}'))
    response = answer_question("Why is this important?", evidence, lambda: provider)
    assert title in response.answer
    assert response.sources[0].excerpts == ([body] if repeats_headline else [title, body])
