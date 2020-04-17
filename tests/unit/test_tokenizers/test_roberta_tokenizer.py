# -*- coding: utf-8 -*-
import os
import unittest

import torch

from caption.tokenizers import RoBERTaTextEncoder
from fairseq.models.roberta import RobertaModel
from test_tube import HyperOptArgumentParser
from torchnlp.download import download_file_maybe_extract

download_file_maybe_extract(
    "https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz",
    directory=os.environ["HOME"] + "/.cache/caption/",
    check_files=["roberta.base/model.pt"],
)
roberta = RobertaModel.from_pretrained(
    os.environ["HOME"] + "/.cache/caption/roberta.base", checkpoint_file="model.pt",
)
original_vocab = roberta.task.source_dictionary.__dict__["indices"]
tokenizer = RoBERTaTextEncoder(roberta.encode, original_vocab)


class TestRoBERTaTextEncoder(unittest.TestCase):
    def test_unk_property(self):
        assert tokenizer.unk_index == original_vocab["<unk>"]

    def test_pad_property(self):
        assert tokenizer.padding_index == original_vocab["<pad>"]

    def test_bos_property(self):
        assert tokenizer.bos_index == original_vocab["<s>"]

    def test_eos_property(self):
        assert tokenizer.eos_index == original_vocab["</s>"]

    def test_mask_property(self):
        assert tokenizer.mask_index == original_vocab["<mask>"]

    def test_vocab_property(self):
        assert isinstance(tokenizer.vocab, dict)

    def test_vocab_size_property(self):
        assert tokenizer.vocab_size == len(original_vocab)

    def test_get_special_tokens_mask(self):
        tensor = torch.tensor([0, 35378, 759, 10269, 56112, 25, 18, 91010, 297, 2])
        mask = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        result = tokenizer.get_special_tokens_mask(tensor)
        assert torch.equal(mask, torch.tensor(result))

        tensor = torch.tensor([0, 35378, 759, 2, 2, 25, 18, 91010, 297, 2])
        mask = torch.tensor([1, 0, 0, 1, 1, 0, 0, 0, 0, 1])
        result = tokenizer.get_special_tokens_mask(tensor)
        assert torch.equal(mask, torch.tensor(result))

    def test_encode(self):
        sentence = "Hello, my dog is cute"
        expected = roberta.encode(sentence)
        result = tokenizer.encode(sentence)
        assert torch.equal(expected, result)
        # Make sure the bos and eos tokens were added.
        assert result[0] == tokenizer.bos_index
        assert result[-1] == tokenizer.eos_index

    def test_batch_encode(self):
        # Test batch_encode.
        batch = ["Hello, my dog is cute", "hello world!"]
        encoded_batch, lengths = tokenizer.batch_encode(batch)

        assert torch.equal(encoded_batch[0], tokenizer.encode(batch[0]))
        assert torch.equal(encoded_batch[1][: lengths[1]], tokenizer.encode(batch[1]))
        assert lengths[0] == len(roberta.encode("Hello, my dog is cute"))
        assert lengths[1] == len(roberta.encode("hello world!"))

        # Check if last sentence is padded.
        assert encoded_batch[1][-1] == tokenizer.padding_index
        assert encoded_batch[1][-2] == tokenizer.padding_index

    def test_encode_trackpos(self):
        sentence = "Hello my dog isn't retarded"
        result = tokenizer.encode_trackpos(sentence)

        # "<s>" "Hello" " my" " dog" " isn"  "'t"  " retarded"  "</s>"
        #   0      1       2     3      4      5        6         7
        expected = (
            torch.tensor([0, 31414, 127, 2335, 965, 75, 47304, 2]),
            torch.tensor([1, 2, 3, 4, 6]),
        )
        assert torch.equal(result[0], expected[0])
        assert torch.equal(result[1], expected[1])

    def test_batch_encode_trackpos(self):
        batch = ["Hello my dog isn't retarded", "retarded"]
        result = tokenizer.batch_encode_trackpos(batch)

        # "<s>" "Hello" " my" " dog" " isn"  "'t"  " retarded"  "</s>"
        #   0      1       2     3      4      5        6         7
        expected1 = (
            torch.tensor([0, 31414, 127, 2335, 965, 75, 47304, 2]),
            torch.tensor([1, 2, 3, 4, 6]),
        )
        # NOTE: since retarded appears in the begin the bpe encoder will
        # not encode it as " retarded" anymore.
        # "</s>" "ret" "arded" "</s>"
        #   0     1      2
        expected2 = (
            torch.tensor([0, 4903, 16230, 2, 1, 1, 1, 1]),
            torch.tensor([1, 0, 0, 0, 0]),
        )
        assert torch.equal(result[0][0], expected1[0])
        assert torch.equal(result[0][1], expected2[0])
        assert torch.equal(result[2][0], expected1[1])
        assert torch.equal(result[2][1], expected2[1])
