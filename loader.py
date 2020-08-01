import os
import torch
import pandas as pd
from logger import get_logger
from vncorenlp import VnCoreNLP
from fairseq.data import Dictionary
from torch.utils.data import DataLoader, Dataset
from fairseq.data.encoders.fastbpe import fastBPE

logger = get_logger('Data Loader')


class BPEConfig:
    bpe_codes = os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'bpe.codes')


class VLSP2016(Dataset):
    def __init__(self, file='SA-2016.train', path=os.path.join('data', 'VLSP2016'), max_length=512):
        super(VLSP2016, self).__init__()
        self.df = pd.read_csv(os.path.join(path, file),
                              names=['sentence', 'label'],
                              sep='\t',
                              encoding='utf-8-sig')

        self.max_length = max_length

        self.bpe = fastBPE(BPEConfig)

        self.vocab = Dictionary()

        self.vocab.add_from_file(os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'dict.txt'))

        self.rdr_segmenter = VnCoreNLP(
            os.path.join('vncorenlp', 'VnCoreNLP-1.1.1.jar'),
            annotators='wseg',
            max_heap_size='-Xmx500m'
        )

        logger.info('Loaded VLSP-2016')

    def __getitem__(self, item):
        text = self.df.iloc[item, 0].encode('utf-8')
        label = self.df.iloc[item, 1]
        text = text.decode('utf-8-sig').strip()
        line = self.rdr_segmenter.tokenize(text)
        line = ' '.join([' '.join(sent) for sent in line])
        subwords = '<s> ' + self.bpe.encode(line) + ' </s>'
        input_ids = self.vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False)

        # temp = torch.zeros((1, self.max_length), dtype=torch.long)
        # temp[0, 0: input_ids.shape[0]] = input_ids

        temp = torch.zeros((self.max_length), dtype=torch.long)

        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]

        temp[0: input_ids.shape[0]] = input_ids
        return temp, 1 if label == 'NEG' else 0

    def __len__(self):
        return len(self.df)


class AIVIVN(Dataset):
    def __init__(self, file='train.crash', path=os.path.join('data', 'AIVIVN')):
        super(AIVIVN, self).__init__()
        with open(os.path.join(path, file), mode='r', encoding='utf-8-sig') as stream:
            self.train = stream.read()

        self.train = self.train.split('\n\ntrain_')
        logger.info('Loaded AIVIVN')

    def __getitem__(self, item):
        sample = self.train[item].strip()
        sent = sample[8:-3].strip()
        label = int(sample[-1:])
        return sent, label

    def __len__(self):
        return len(self.train)


class UITVSFC(Dataset):
    def __init__(self, file='train', path=os.path.join('data', 'UIT-VSFC')):
        super(UITVSFC, self).__init__()
        with open(os.path.join(path, file, 'sents.txt'), mode='r', encoding='utf-8-sig') as stream:
            self.sents = stream.read().strip().split('\n')
        with open(os.path.join(path, file, 'sentiments.txt'), mode='r', encoding='utf-8-sig') as stream:
            self.sentiments = stream.read().strip().split('\n')
        with open(os.path.join(path, file, 'topics.txt'), mode='r', encoding='utf-8-sig') as stream:
            self.topics = stream.read().strip().split('\n')

        logger.info('Loaded UIT-VSFC')

    def __getitem__(self, item):
        return self.sents[item].strip(), int(self.sentiments[item])

    def __len__(self):
        return len(self.sents)