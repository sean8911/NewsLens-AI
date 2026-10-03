"""Serialize untrusted prompt fields without literal chat-template delimiters."""

import json


def untrusted_json(value) -> str:
    # The local Qwen tokenizer recognizes tokens such as <|im_start|>. JSON escaping
    # preserves the original text on decoding, without inserting those literal tokens
    # into the model's chat template. This is structural protection, not a claim that
    # natural-language prompt injection is solved; output validation remains required.
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
