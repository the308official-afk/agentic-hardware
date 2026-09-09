"""Host configuration; constructing identity never imports a tokenizer."""
from __future__ import annotations

import json
from pathlib import Path
from .models import CodecConfig
from .pipeline import PromptEncoder


def load_encoder(path: str | Path | None, model: str = "", *, allow_tokenizer_download: bool = False) -> PromptEncoder:
    if not path:
        return PromptEncoder()
    raw = json.loads(Path(path).read_text())
    config = CodecConfig(**raw.get("codec", {}))
    if config.codec == "identity":
        return PromptEncoder(config)
    from .tokenizers import HuggingFaceTokenCounter
    options = dict(raw.get("tokenizer", {}))
    if allow_tokenizer_download:
        options["local_files_only"] = False
    options.setdefault("model", model)
    if not options["model"]:
        raise ValueError("tokenizer model is required")
    if model and options["model"] != model:
        raise ValueError("configured tokenizer model differs from the served model")
    # Configuration errors stop startup. Runtime per-request failures bypass.
    return PromptEncoder(config, HuggingFaceTokenCounter(**options))
