import os
import torch
from transformers import BertTokenizer
from vncorenlp import VnCoreNLP
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE


class BPEConfig:
    bpe_codes = os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'bpe.codes')


class PhoBertTokenizer:
    def __init__(self, max_length=512):
        self.bpe = fastBPE(BPEConfig)
        self.vocab = Dictionary()
        self.vocab.add_from_file(os.path.join(os.getcwd(),
                                              'pretrained',
                                              'PhoBERT_base_transformers',
                                              'dict.txt'))
        self.rdr_segmenter = VnCoreNLP(
            os.path.join('vncorenlp', 'VnCoreNLP-1.1.1.jar'),
            annotators='wseg',
            max_heap_size='-Xmx500m'
        )
        self.max_length = max_length

    def __call__(self, x):
        line = self.rdr_segmenter.tokenize(x)
        line = ' '.join([' '.join(sent) for sent in line])
        subwords = '<s> ' + self.bpe.encode(line) + ' </s>'
        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False)
        temp = torch.zeros(self.max_length, dtype=torch.long)
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
        temp[0: input_ids.shape[0]] = input_ids
        return temp


class BertViTokenizer:
    def __init__(self, max_length=512, shortcut_pretrained='bert-base-multilingual-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(shortcut_pretrained)

    def __call__(self, x):
        return torch.tensor([self.tokenizer.encode(x, add_special_tokens=True)])
