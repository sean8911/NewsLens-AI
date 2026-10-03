import json

import pytest

from nlp.summarization.cli import main
from nlp.summarization.summarizer import MAX_TEXT, ArticleSummarizer

BODY = (
    "Dr. Smith said the trial did not prove a cure. "
    "The trial involved 1,200 adults across three hospitals. "
    "Researchers will continue monitoring participants for another year to assess safety. "
    "The results were presented at a scientific conference attended by doctors and researchers. "
    "The team plans to publish its complete findings after the follow-up work has finished."
)


def test_complete_verbatim_sentences_preserve_attribution_negation_and_numbers():
    result = ArticleSummarizer().summarize("Trial findings", content=BODY, language="en")
    assert result.summary_status == "summarized"
    assert result.summary_one_sentence == "Dr. Smith said the trial did not prove a cure."
    assert "1,200 adults" in result.summary
    assert len(result.summary) <= min(900, int(len(BODY) * 0.6))
    assert len(result.summary_one_sentence) <= 500
    assert result.summary_created_at.tzinfo is not None
    assert result.summary_bullets and " ".join(result.summary_bullets) == result.summary
    selected = [BODY[span["start"] : span["end"]] for span in result.summary_sentence_spans]
    assert selected == result.summary_bullets
    assert result.summary == " ".join(selected)
    assert result.summary_text_source == "content"
    assert isinstance(result.values(json_mode=True)["summary_created_at"], str)


def test_description_fallback_and_content_precedence():
    summarizer = ArticleSummarizer()
    fallback = summarizer.summarize("Trial findings", description=BODY, language="en")
    assert fallback.summary_text_source == "description"
    result = summarizer.summarize("Trial findings", BODY, "A short body.", "en")
    assert result.summary is None and result.summary_text_source == "content"
    assert summarizer.summarize(BODY, language="en").summary_status == "insufficient_text"


@pytest.mark.parametrize(
    "language,status",
    [(None, "unknown_language"), ("ms", "unsupported_language"), ("zh", "unsupported_language")],
)
def test_language_status(language, status):
    result = ArticleSummarizer().summarize("Trial findings", BODY, language=language)
    assert result.summary_status == status and result.summary is None


@pytest.mark.parametrize(
    "body",
    [
        "Researchers continue monitoring the participants to assess long-term safety "
        "and whether treatment improves their health.",
        "An incomplete lead with no terminal punctuation\n" + BODY,
        "What makes a treatment safe and effective? " + BODY,
        'The agency said, "The water is not safe. Stay indoors." ' + BODY,
        "The study found a promising treatment for the disease. "
        "However, the results did not prove a cure. " + BODY,
        "A very long lead " + "containing many source details " * 30 + ". " + BODY,
    ],
)
def test_abstains_instead_of_cutting_clauses_or_omitting_obvious_caveats(body):
    result = ArticleSummarizer().summarize("Trial findings", content=body, language="en")
    assert result.summary_status == "not_compressible"
    assert result.summary is None and result.summary_one_sentence is None
    assert result.summary_created_at is None and result.summary_sentence_spans == []


def test_hash_changes_with_body_language_and_source_and_clipping_is_explicit():
    summarizer = ArticleSummarizer()
    a = summarizer.prepare("Title", BODY, None, "en")
    assert a[2] != summarizer.prepare("Title", None, BODY, "en")[2]
    assert a[2] != summarizer.prepare("Title", BODY + " More.", None, "en")[2]
    assert a[2] != summarizer.prepare("Title", BODY, None, "ms")[2]
    result = summarizer.summarize("Title", content=BODY * 150, language="en")
    assert result.summary_status == "summarized" and result.summary_text_truncated
    assert result.summary_source_chars == MAX_TEXT


def test_exact_offsets_preserve_whitespace_inside_sentences():
    body = "  Doctors reported  no evidence of a cure.\n\n" + BODY
    result = ArticleSummarizer().summarize("Trial findings", content=body, language="en")
    assert result.summary_one_sentence == "Doctors reported  no evidence of a cure."
    assert result.summary_sentence_spans[0]["start"] == 2


@pytest.mark.parametrize("invented", [True, False])
def test_backend_errors_do_not_return_ungrounded_text_or_leak_messages(invented, caplog):
    class Backend:
        version = "fixture/v1"
        languages = frozenset({"en"})

        def split(self, text):
            if invented:
                return ["The trial proved a cure.", "Everyone recovered."]
            raise RuntimeError("secret diagnostic")

    result = ArticleSummarizer(Backend()).summarize("Trial findings", content=BODY, language="en")
    assert result.summary_status == "error" and result.summary is None
    assert "secret diagnostic" not in caplog.text


def test_cli_emits_serializable_json_without_database(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "nlp.summarization.cli.create_database_engine", lambda *_: pytest.fail("Database accessed")
    )
    path = tmp_path / "article.txt"
    path.write_text(BODY, encoding="utf-8")
    assert main(["summarize", "--title", "Trial findings", "--content-file", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["summary_status"] == "summarized" and result["summary_created_at"]


def test_cli_missing_file_and_invalid_batch(tmp_path):
    assert (
        main(
            [
                "summarize",
                "--title",
                "Trial findings",
                "--content-file",
                str(tmp_path / "missing.txt"),
            ]
        )
        == 1
    )
    with pytest.raises(SystemExit) as exc:
        main(["backfill", "--batch-size", "0"])
    assert exc.value.code == 2


def test_single_quotes_are_not_cut_and_contractions_are_preserved():
    result = ArticleSummarizer().summarize(
        "Trial findings",
        content="The doctor said, 'This isn't a cure. More work is needed.' " + BODY,
        language="en",
    )
    assert result.summary_status == "not_compressible"
    result = ArticleSummarizer().summarize(
        "Trial findings",
        content="The treatment didn't prove effective in the trial. " + BODY,
        language="en",
    )
    assert result.summary_one_sentence == "The treatment didn't prove effective in the trial."
