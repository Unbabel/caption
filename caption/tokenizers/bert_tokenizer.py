# -*- coding: utf-8 -*-
r"""
Hugging Face BERT Tokenizer class
==============
    Hugging Face BERT tokenizer wrapper.
"""
import torch

from .tokenizer_base import TextEncoderBase
from torchnlp.encoders.text.text_encoder import TextEncoder
from transformers import BertTokenizer


class BERTTextEncoder(TextEncoderBase):
    """
    BERT tokenizer.

    :param model: BERT model to be used.
    """

    def __init__(self, model: str) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model)
        # Properties from the base class
        self.stoi = self.tokenizer.vocab
        self.itos = self.tokenizer.ids_to_tokens
        self._bos_index = self.tokenizer.cls_token_id
        self._pad_index = self.tokenizer.pad_token_id
        self._eos_index = self.tokenizer.sep_token_id
        self._unk_index = self.tokenizer.unk_token_id
        self._mask_index = self.tokenizer.mask_token_id

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
