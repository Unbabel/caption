# -*- coding: utf-8 -*-
from .bert import BERT
from .roberta import RoBERTa
from .xlm_roberta import XLMRoBERTa
from .encoder_base import Encoder
from .hf_roberta import HuggingFaceRoBERTa


str2encoder = {
    "BERT": BERT,
    "RoBERTa": RoBERTa,
    "XLM-RoBERTa": XLMRoBERTa,
    "HF-RoBERTa": HuggingFaceRoBERTa,  # legacy
}

__all__ = [
    "Encoder",
    "BERT",
    "RoBERTa",
    "XLMRoBERTa",
    "HuggingFaceRoBERTa",  # legacy
]
