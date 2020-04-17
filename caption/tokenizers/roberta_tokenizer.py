# -*- coding: utf-8 -*-
r"""
RoBERTa Tokenizer class
==============
    Fairseq RoBERTa tokenizer wrapper.
"""
import torch

from .tokenizer_base import TextEncoderBase
from torchnlp.encoders.text.text_encoder import TextEncoder


class RoBERTaTextEncoder(TextEncoderBase):
    """
    RoBERTa encoder from Fairseq.

    :param tokenizer_func: RoBERTa tokenization function.
        This can be easily obtain from the fairseq model (e.g: roberta.encode callable)
    :param vocabulary: the dictionary containing the RoBERTa vocabulary. 
        This can be easily obtain from the fairseq model 
        (e.g: roberta.task.source_dictionary.__dict__['indices'])
    """

    def __init__(self, encode_func: callable, vocabulary: dict) -> None:
        super().__init__()

        self.encode_func = encode_func
        # Properties from the base class
        self.stoi = vocabulary
        self.itos = {v: k for k, v in vocabulary.items()}
        self._pad_index = self.stoi["<pad>"]
        self._eos_index = self.stoi["</s>"]
        self._unk_index = self.stoi["<unk>"]
        self._bos_index = self.stoi["<s>"]
        self._mask_index = self.stoi["<mask>"]

    def encode_trackpos(self, sequence: str) -> torch.Tensor:
        """ Encodes a 'sequence' and keeps the alignments with the respective tags.
        :param sequence: String 'sequence' to encode.

        Returns:
            - torch.Tensor: Encoding of the 'sequence'.
            - torch.Tensor: Alignment indexes
        """
        sequence = TextEncoder.encode(self, sequence)
        tag_index, vector = [], [self._bos_index,]
        tokens = sequence.split()
        # Add whitespace to each token to prevent Ġ<token>
        tokens = [tokens[0]] + [" " + token for token in tokens[1:]]
        for index, token in enumerate(tokens):
            tag_index.append(len(vector))
            vector = vector + self.encode_func(token)[1:-1].tolist()
        vector.append(self._eos_index)
        return torch.tensor(vector), torch.tensor(tag_index)

    def encode(self, sequence: str) -> torch.Tensor:
        """ Encodes a 'sequence'.
        :param sequence: String 'sequence' to encode.
        
        Returns:
            - torch.Tensor: Encoding of the 'sequence'.
        """
        sequence = TextEncoder.encode(self, sequence)
        return self.encode_func(sequence)
