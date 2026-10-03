import json
from functools import lru_cache
from importlib.metadata import version
from threading import Lock

from backend.app.core.config import Settings
from backend.app.core.performance import measured, record_tokens
from backend.app.services.chat_context import POLICY, SYSTEM_PROMPT
from nlp.embeddings.service import digest


class LocalLanguageModel:
    def __init__(self, settings: Settings):
        self.model_id = settings.chat_model_id
        self.revision = settings.chat_model_revision
        self.path = settings.chat_model_path / digest([self.model_id, self.revision])[:20]
        self.max_input_tokens = settings.chat_max_input_tokens
        self.max_seconds = settings.chat_generation_seconds
        self.version = digest(
            [
                self.model_id,
                self.revision,
                POLICY,
                SYSTEM_PROMPT,
                "cpu-float32-greedy-128-new-tokens",
                self.max_input_tokens,
                self.max_seconds,
                version("transformers"),
                version("torch"),
            ]
        )

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        self.path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.model_id,
            revision=self.revision,
            local_dir=str(self.path),
            allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors"],
            ignore_patterns=["onnx/*", "openvino/*"],
        )
        _load(str(self.path.resolve()))
        (self.path / "newslens-ready.json").write_text(
            json.dumps({"model_id": self.model_id, "revision": self.revision}), encoding="utf-8"
        )

    @measured("generation")
    def generate(self, messages: list[dict]) -> str:
        marker = self.path / "newslens-ready.json"
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != {
            "model_id": self.model_id,
            "revision": self.revision,
        }:
            raise RuntimeError("Chat model unavailable; run newslens-chat download")
        import torch

        model, tokenizer, lock = _load(str(self.path.resolve()))
        with lock, torch.inference_mode():
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            length = inputs["input_ids"].shape[-1]
            if (
                length > self.max_input_tokens
                or length + 128 > model.config.max_position_embeddings
            ):
                raise ValueError("Chat context exceeds the model input budget")
            record_tokens("llm", length, calls=1)
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                max_time=self.max_seconds,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            record_tokens("llm", output_tokens=len(outputs[0][length:]), completed=1)
            return tokenizer.decode(outputs[0][length:], skip_special_tokens=True).strip()

    def context_fits(self, messages: list[dict]) -> bool:
        # A character count is not a token budget, especially for Chinese evidence.
        tokenizer = _tokenizer(str(self.path.resolve()))
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True
        )
        return len(encoded["input_ids"]) <= self.max_input_tokens


@lru_cache(maxsize=1)
def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


@lru_cache(maxsize=1)
def _load(path: str):
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(2)
    tokenizer = _tokenizer(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
    )
    model.eval()
    return model, tokenizer, Lock()


def language_model_for(settings: Settings) -> LocalLanguageModel:
    return LocalLanguageModel(settings)
