# -*- coding: utf-8 -*-
r"""
Hugging Face RoBERTa implementation.
==============
    RoBERTa model from hugging face transformers repo.
"""
import torch
from transformers import RobertaModel, RobertaForMaskedLM

from caption.models.encoders.encoder_base import Encoder
from caption.tokenizers import HfRoBERTaTextEncoder
from test_tube import HyperOptArgumentParser
from torchnlp.utils import lengths_to_mask


class HuggingFaceRoBERTa(Encoder):
    """
    Hugging Face RoBERTa encoder.
    
    :param tokenizer: RoBERTa text encoder.
    :param hparams: HyperOptArgumentParser obj.
    :param lm_head: If true the language model head from the pretrain model is saved. 

    Check the available models here: 
        https://huggingface.co/transformers/pretrained_models.html
    """

    def __init__(
        self,
        tokenizer: HfRoBERTaTextEncoder,
        hparams: HyperOptArgumentParser,
        lm_head: bool = False,
    ) -> None:
        super().__init__(768 if "base" in hparams.pretrained_model else 1024, tokenizer)
        self._n_layers = 13 if "base" in hparams.pretrained_model else 25
        self.padding_idx = self.tokenizer.padding_index
        if not lm_head:
            self.model = RobertaModel.from_pretrained(
                hparams.pretrained_model, output_hidden_states=True
            )
        else:
            mlm_model = RobertaForMaskedLM.from_pretrained(
                hparams.pretrained_model, output_hidden_states=True
            )
            self.model = mlm_model.roberta
            self.lm_head = mlm_model.lm_head

    @classmethod
    def from_pretrained(cls, hparams: HyperOptArgumentParser, lm_head: bool = False):
        """ Function that loads a pretrained RoBERTa encoder.
        :param hparams: HyperOptArgumentParser obj.
        
        Returns:
            - RoBERTa Encoder model from hugging face
        """
        tokenizer = HfRoBERTaTextEncoder(model=hparams.pretrained_model)
        model = RoBERTa(tokenizer=tokenizer, hparams=hparams, lm_head=lm_head)
        return model

    def forward(self, tokens: torch.tensor, lengths: torch.tensor, **kwargs) -> dict:
        """
        Encodes a batch of sequences.
        :param tokens: Torch tensor with the input sequences [batch_size x seq_len].
        :param lengths: Torch tensor with the length of each sequence [seq_len].

        Returns: 
            - 'sentemb': tensor [batch_size x 1024] with the sentence encoding.
            - 'wordemb': tensor [batch_size x seq_len x 1024] with the word level embeddings.
            - 'all_layers': List with the word_embeddings returned by each layer.
            - 'mask': torch.Tensor [seq_len x batch_size] 
            - 'extra': tuple with the last_hidden_state [batch_size x seq_len x hidden_size],
                the pooler_output representing the entire sentence and the word embeddings for 
                all XLM-R layers (list of tensors [batch_size x seq_len x hidden_size]) 
        """
        mask = lengths_to_mask(lengths, device=tokens.device)
        # Run  RoBERTa model.
        last_hidden_states, pooler_output, all_layers = self.model(tokens, mask)
        return {
            "sentemb": pooler_output,
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "mask": mask,
            "extra": (last_hidden_states, pooler_output, all_layers),
        }
