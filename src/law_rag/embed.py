"""BGE-M3 embedding wrapper. Singleton model load."""
from __future__ import annotations

from typing import List

import numpy as np

from . import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel

        _model = BGEM3FlagModel(config.EMBED_MODEL, use_fp16=False)
    return _model


def encode(texts: List[str]) -> np.ndarray:
    """Return dense embeddings of shape (n, 1024) for the given texts."""
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    model = _get_model()
    out = model.encode(texts, batch_size=8, max_length=2048, return_dense=True)
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    return dense
