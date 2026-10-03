import json
from functools import lru_cache
from importlib.metadata import version
from threading import Lock
from time import monotonic

from nlp.embeddings.service import digest
from nlp.multilingual.glossary import PLACES, prepare_terms


class LocalTranslator:
    """Many-to-many translation with explicit source/target language tokens, no chat prompts."""

    def __init__(self, settings):
        self.model_id = settings.translation_model_id
        self.revision = settings.translation_model_revision
        self.path = settings.translation_model_path / digest([self.model_id, self.revision])[:20]
        self.max_seconds = settings.chat_generation_seconds
        self.version = digest(
            [
                self.model_id,
                self.revision,
                "m2m100-cpu-beam5-384-v1",
                PLACES,
                self.max_seconds,
                version("transformers"),
                version("torch"),
            ]
        )

    def download(self):
        from huggingface_hub import snapshot_download
        from transformers import M2M100ForConditionalGeneration

        self.path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.model_id,
            revision=self.revision,
            local_dir=str(self.path),
            allow_patterns=["*.json", "*.model", "pytorch_model.bin"],
        )
        # Official pinned checkpoint; restricted weights-only loading (PyTorch >=2.6).
        # Convert once to safetensors, which is the only format loaded during inference.
        if not (self.path / "model.safetensors").is_file():
            model = M2M100ForConditionalGeneration.from_pretrained(
                str(self.path.resolve()),
                local_files_only=True,
                use_safetensors=False,
                weights_only=True,
            )
            model.save_pretrained(self.path, safe_serialization=True)
            del model
        _load(str(self.path.resolve()))
        (self.path / "newslens-ready.json").write_text(
            json.dumps({"model_id": self.model_id, "revision": self.revision}), encoding="utf-8"
        )

    def translate(self, text: str, source: str, target: str) -> str:
        if source not in {"en", "ms", "zh"} or target not in {"en", "ms", "zh"}:
            raise ValueError("Unsupported translation language")
        if not text.strip() or len(text) > 500:
            raise ValueError("Translation requires a bounded excerpt")
        marker = self.path / "newslens-ready.json"
        if not marker.is_file() or json.loads(marker.read_text("utf-8")) != {
            "model_id": self.model_id,
            "revision": self.revision,
        }:
            raise RuntimeError("Translation model unavailable; run newslens-multilingual download")
        import torch

        model, tokenizer, lock = _load(str(self.path.resolve()))
        with lock, torch.inference_mode():
            tokenizer.src_lang = source
            inputs = tokenizer(prepare_terms(text, target), return_tensors="pt", truncation=False)
            if inputs["input_ids"].shape[-1] > 512:
                raise ValueError("Translation input token budget exceeded")
            started = monotonic()
            output = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(target),
                max_new_tokens=384,
                max_length=None,
                num_beams=5,
                do_sample=False,
                max_time=self.max_seconds,
            )
            if (
                output[0][-1].item() != tokenizer.eos_token_id
                or monotonic() - started >= self.max_seconds
                or output.shape[-1] >= 385
            ):
                raise ValueError("Incomplete translation")
            return tokenizer.decode(output[0], skip_special_tokens=True).strip()


@lru_cache(maxsize=1)
def _load(path):
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    torch.set_num_threads(2)
    tokenizer = M2M100Tokenizer.from_pretrained(path, local_files_only=True)
    model = M2M100ForConditionalGeneration.from_pretrained(
        path,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
    )
    model.eval()
    return model, tokenizer, Lock()
