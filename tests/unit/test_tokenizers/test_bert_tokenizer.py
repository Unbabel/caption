# -*- coding: utf-8 -*-
import unittest

import torch
from transformers import BertTokenizer

from caption.tokenizers import BERTTextEncoder


class TestBERTTextEncoder(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BERTTextEncoder("bert-base-multilingual-cased")
        self.original_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-multilingual-cased"
        )

    def test_unk_property(self):
        assert self.tokenizer.unk_index == 100

    def test_pad_property(self):
        assert self.tokenizer.padding_index == 0

    def test_bos_property(self):
        assert self.tokenizer.bos_index == 101

    def test_eos_property(self):
        assert self.tokenizer.eos_index == 102

    def test_mask_property(self):
        assert self.tokenizer.mask_index == 103

    def test_vocab_property(self):
        assert isinstance(self.tokenizer.vocab, dict)

    def test_vocab_size_property(self):
        assert self.tokenizer.vocab_size > 0

    def test_get_special_tokens_mask(self):
        tensor = torch.tensor([101, 35378, 759, 10269, 56112, 25, 18, 91010, 297, 102])
        mask = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        result = self.tokenizer.get_special_tokens_mask(tensor)
        assert torch.equal(mask, torch.tensor(result))

        tensor = torch.tensor([101, 35378, 759, 102, 25, 18, 91010, 297, 102])
        mask = torch.tensor([1, 0, 0, 1, 0, 0, 0, 0, 1])
        result = self.tokenizer.get_special_tokens_mask(tensor)
        assert torch.equal(mask, torch.tensor(result))

    def test_encode(self):
        sentence = "Hello, my dog is cute"
        expected = self.original_tokenizer.encode(sentence)
        result = self.tokenizer.encode(sentence)
        assert torch.equal(torch.tensor(expected), result)
        # Make sure the bos and eos tokens were added.
        assert result[0] == self.tokenizer.bos_index
        assert result[-1] == self.tokenizer.eos_index

    def test_batch_encode(self):
        # Test batch_encode.
        batch = ["Hello, my dog is cute", "hello world!"]
        encoded_batch, lengths = self.tokenizer.batch_encode(batch)

        assert torch.equal(encoded_batch[0], self.tokenizer.encode(batch[0]))
        assert torch.equal(
            encoded_batch[1][: lengths[1]], self.tokenizer.encode(batch[1])
        )
        assert lengths[0] == len(
            self.original_tokenizer.encode("Hello, my dog is cute")
        )
        assert lengths[1] == len(self.original_tokenizer.encode("hello world!"))

        # Check if last sentence is padded.
        assert encoded_batch[1][-1] == self.tokenizer.padding_index
        assert encoded_batch[1][-2] == self.tokenizer.padding_index

    def test_encode_trackpos(self):
        sentence = "Hello my dog isn't retarded"
        result = self.tokenizer.encode_trackpos(sentence)

        # [CLS] hello  my  dog   isn   '  t  ret  ##arde  ##d  [SEP]"
        #   0    1     2   3     4     5  6   7      8     9
        expected = (
            torch.tensor(
                [101, 31178, 15127, 17835, 98370, 112, 188, 62893, 45592, 10162, 102]
            ),
            torch.tensor([1, 2, 3, 4, 7]),
        )
        assert torch.equal(result[0], expected[0])
        assert torch.equal(result[1], expected[1])

    def test_batch_encode_trackpos(self):
        batch = ["Hello my dog isn't retarded", "retarded"]
        result = self.tokenizer.batch_encode_trackpos(batch)

        # [CLS] hello  my  dog   isn   '  t  ret  ##arde  ##d  [SEP]"
        #   0    1     2   3     4     5  6   7      8     9
        expected1 = (
            torch.tensor(
                [101, 31178, 15127, 17835, 98370, 112, 188, 62893, 45592, 10162, 102]
            ),
            torch.tensor([1, 2, 3, 4, 7]),
        )

        # [CLS] ret  ##arde  ##d  [SEP]"
        #   0    1     2      3
        expected2 = (
            torch.tensor([101, 62893, 45592, 10162, 102, 0, 0, 0, 0, 0, 0]),
            torch.tensor([1, 0, 0, 0, 0]),
        )

        assert torch.equal(result[0][0], expected1[0])
        assert torch.equal(result[0][1], expected2[0])
        assert torch.equal(result[2][0], expected1[1])
        assert torch.equal(result[2][1], expected2[1])
