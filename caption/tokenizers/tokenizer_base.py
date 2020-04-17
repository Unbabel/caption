# -*- coding: utf-8 -*-
r"""
Tokenizer base class
==============
    Base class for all tokenizers.
"""
import torch

from torchnlp.encoders import Encoder
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.encoders.text.text_encoder import TextEncoder


class TextEncoderBase(TextEncoder):
    """
    Base class for the specific tokenizers of each model.
    """

    def __init__(self) -> None:
        self.enforce_reversible = False

    @property
    def unk_index(self) -> int:
        """ Returns the index used for the unknown token. """
        return self._unk_index

    @property
    def bos_index(self) -> int:
        """ Returns the index used for the begin-of-sentence token. """
        return self._bos_index

    @property
    def eos_index(self) -> int:
        """ Returns the index used for the end-of-sentence token. """
        return self._eos_index

    @property
    def mask_index(self) -> int:
        """ Returns the index used for the end-of-sentence token. """
        return self._mask_index

    @property
    def padding_index(self) -> int:
        """ Returns the index used for padding. """
        return self._pad_index

    @property
    def vocab(self) -> list:
        """
        Returns:
            list: List of tokens in the dictionary.
        """
        return self.stoi

    @property
    def vocab_size(self) -> int:
        """
        Returns:
            int: Number of tokens in the dictionary.
        """
        return len(self.itos)

    def tokenize(self, sequence: str) -> list:
        """ 
        Function that tokenizes a string. 
        - To be extended by subclasses. 
        """
        raise NotImplementedError

    def encode(self, sequence: str) -> torch.Tensor:
        """ Encodes a 'sequence'.
        :param sequence: String 'sequence' to encode.
        
        Returns:
            - torch.Tensor: Encoding of the 'sequence'.
        """
        sequence = super().encode(sequence)
        tokens = self.tokenize(sequence)
        vector = [self.stoi.get(token, self.unk_index) for token in tokens]
        return torch.tensor(vector)

    def batch_encode(self, iterator, dim=0, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        :param iterator (iterator): Batch of text to encode.
        :param dim (int, optional): Dimension along which to concatenate tensors.
        :param **kwargs: Keyword arguments passed to 'encode'.
            
        Returns
            torch.Tensor, torch.Tensor: Encoded and padded batch of sequences; Original lengths of
                sequences.
        """
        return stack_and_pad_tensors(
            Encoder.batch_encode(self, iterator, **kwargs),
            padding_index=self.padding_index,
            dim=dim,
        )

    def encode_trackpos(self, sequence: str) -> torch.Tensor:
        """ Encodes a 'sequence' and keeps track of the position of the beginning of each token before wordpieces.
        :param sequence: String 'sequence' to encode.

        Returns:
            - torch.Tensor: Encoding of the 'sequence'.
            - torch.Tensor: Alignment indexes
        """
        raise NotImplementedError

    def batch_encode_trackpos(
        self, iterator, dim=0, **kwargs
    ) -> (torch.Tensor, torch.Tensor):
        """
        :param iterator (iterator): Batch of text to encode.
        :param dim (int, optional): Dimension along which to concatenate tensors.
        :param **kwargs: Keyword arguments passed to 'encode'.
            
        Returns
            torch.Tensor, torch.Tensor: Encoded and padded batch of sequences; Original lengths of
                sequences.
        """
        sequences, tags = zip(*[self.encode_trackpos(object_) for object_ in iterator])
        sequences, seq_lengths = stack_and_pad_tensors(
            sequences, padding_index=self.padding_index, dim=dim
        )
        tag_idxs, tag_lengths = stack_and_pad_tensors(tags, padding_index=0, dim=dim)
        return sequences, seq_lengths, tag_idxs, tag_lengths

    def get_special_tokens_mask(self, tokens):
        """ Function from Hugging face to train language models. """
        return list(
            map(lambda x: 1 if x in [self.bos_index, self.eos_index] else 0, tokens)
        )
