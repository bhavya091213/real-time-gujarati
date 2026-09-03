"""Compatibility shims so IndicTrans2 remote code + IndicTransToolkit load
under transformers 5.x.

Import this module BEFORE importing IndicTransToolkit or loading the
IndicTrans2 model. Two breakages are papered over:

- `transformers.onnx` was removed in v5; the IndicTrans2 configuration file
  imports OnnxConfig classes it never actually uses at inference time.
- `PreTrainedTokenizerBase` moved out of `transformers.tokenization_utils`;
  IndicTransToolkit's collator imports it from the old location.
"""

import sys
import types

import transformers.tokenization_utils as _tokenization_utils
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if "transformers.onnx" not in sys.modules:
    onnx_mod = types.ModuleType("transformers.onnx")
    onnx_mod.__path__ = []  # mark as package so submodule imports resolve
    onnx_mod.OnnxConfig = type("OnnxConfig", (), {})
    onnx_mod.OnnxSeq2SeqConfigWithPast = type("OnnxSeq2SeqConfigWithPast", (), {})

    onnx_utils = types.ModuleType("transformers.onnx.utils")
    onnx_utils.compute_effective_axis_dimension = lambda dim, *a, **k: dim
    onnx_mod.utils = onnx_utils

    sys.modules["transformers.onnx"] = onnx_mod
    sys.modules["transformers.onnx.utils"] = onnx_utils

if not hasattr(_tokenization_utils, "PreTrainedTokenizerBase"):
    _tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
