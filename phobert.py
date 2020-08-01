import os
import json
import torch

from vncorenlp import VnCoreNLP
from fairseq.data import Dictionary
from transformers import RobertaModel, RobertaConfig
from fairseq.data.encoders.fastbpe import fastBPE


class BPEConfig:
    bpe_codes = os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'bpe.codes')
    

class PhoBertEncoder(torch.nn.Module):
    def __init__(self):
        super(PhoBertEncoder, self).__init__()

        self.config = RobertaConfig.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'config.json')
        )

        self.phobert = RobertaModel.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'model.bin'),
            config=self.config
        )

        self.rdr_segmenter = VnCoreNLP(
            os.path.join('vncorenlp', 'VnCoreNLP-1.1.1.jar'),
            annotators='wseg',
            max_heap_size='-Xmx500m'
        )

        self.bpe = fastBPE(BPEConfig)

        self.vocab = Dictionary()

        self.vocab.add_from_file(os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'dict.txt'))

    def __call__(self, text):
        line = self.rdr_segmenter.tokenize(text)
        subwords = '<s> ' + self.bpe.encode(line) + ' </s>'
        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long.tolist()
        all_input_ids = torch.tensor([input_ids], dtype=torch.long)

        features = self.phobert(all_input_ids)
        return features


