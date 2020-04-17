# -*- coding: utf-8 -*-
r"""
Hugging Face RoBERTa Tokenizer class
==============
    Hugging Face RoBERTa tokenizer wrapper.
"""
import torch

from .tokenizer_base import TextEncoderBase
from torchnlp.encoders.text.text_encoder import TextEncoder
from transformers import RobertaTokenizer


class HfRoBERTaTextEncoder(TextEncoderBase):
    """
    Hugging Face RoBERTa tokenizer.

    :param model: RoBERTa model to be used.
    """

    def __init__(self, model: str) -> None:
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model)
        # Properties from the base class
        self.stoi = {}  # ignored
        self.itos = {}  # ignored
        self._bos_index = self.tokenizer.bos_token_id
        self._pad_index = self.tokenizer.pad_token_id
        self._eos_index = self.tokenizer.sep_token_id
        self._unk_index = self.tokenizer.unk_token_id
        # Hugging face mask_token_id is wrong! this is the real value:
        self._mask_index = 250001
        # TODO update transformers version to avoid the above bug.
        # https://github.com/huggingface/transformers/pull/2509

    @property
    def vocab_size(self) -> int:
        """
        Returns:
            int: Number of tokens in the dictionary.
        """
        return self.tokenizer.vocab_size

    def encode_trackpos(self, sequence: str) -> torch.Tensor:
        """ Encodes a 'sequence' and keeps the alignments with the respective tags.
        :param sequence: String 'sequence' to encode.

        Returns:
            - torch.Tensor: Encoding of the 'sequence'.
            - torch.Tensor: Alignment indexes
        """
        sequence = TextEncoder.encode(self, sequence)
        tag_index, vector = [], [self._bos_index,]
        for index, token in enumerate(sequence.split()):
            tag_index.append(len(vector))
            vector = vector + self.tokenizer.encode(token, add_special_tokens=False)
        vector.append(self._eos_index)
        return torch.tensor(vector), torch.tensor(tag_index)

    def encode(self, sequence: str) -> torch.Tensor:
        """ Encodes a 'sequence'.
        :param sequence: String 'sequence' to encode.
        
        Returns:
            - torch.Tensor: Encoding of the 'sequence'.
        """
        sequence = TextEncoder.encode(self, sequence)
        vector = self.tokenizer.encode(sequence)
        return torch.tensor(vector)
