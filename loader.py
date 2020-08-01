import os
import re
import pandas as pd
from logger import get_logger
from torch.utils.data import DataLoader, Dataset

logger = get_logger('Data Loader')


class VLSP2016(Dataset):
    def __init__(self, file='SA-train.train', path=os.path.join('data', 'VLSP2016')):
        super(VLSP2016, self).__init__()
        self.df = pd.read_csv(os.path.join(path, file),
                              names=['sentence', 'label'],
                              sep='\t',
                              encoding='utf-8-sig')
        logger.info('Loaded VLSP-2016')

    def __getitem__(self, item):
        sent = self.df.iloc[item, 0].encode('utf-8')
        label = self.df.iloc[item, 1]
        return sent.decode('utf-8-sig').strip(), 1 if label == 'NEG' else 0

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

