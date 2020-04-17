# -*- coding: utf-8 -*-
import torch
from caption.tokenizers import TextEncoderBase


def mask_fill(
    fill_value: float,
    tokens: torch.tensor,
    embeddings: torch.tensor,
    padding_index: int,
) -> torch.tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)


def mask_tokens(
    inputs: torch.tensor,
    tokenizer: TextEncoderBase,
    mlm_probability: float = 0.15,
    ignore_index: int = -100,
):
    """ Mask tokens function from Hugging Face that prepares masked tokens inputs/labels for 
    masked language modeling.

    :param inputs: Input tensor to be masked.
    :param tokenizer: COMET text encoder.
    :param mlm_probability: Probability of masking a token (default: 15%).
    :param ignore_index: Specifies a target value that is ignored and does not contribute to 
        the input gradient (default: -100).

    Returns:
        - Tuple with input to the model and the target.
    """
    if tokenizer.mask_index is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language"
            "modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )
    padding_mask = labels.eq(tokenizer.padding_index)
    probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = ignore_index  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )

    inputs[indices_replaced] = tokenizer.mask_index

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
