import os
import torch
import pandas as pd
from utils.logger import get_logger
from torch.utils.data import Dataset
from utils.tokenizer import PhoBertTokenizer, BertViTokenizer

logger = get_logger('Data Loader')
BERTvi = ['phobert', 'bert-base-multilingual-cased']


class VLSP2016(Dataset):
    def __init__(self,
                 file='SA-2016.train',
                 path=os.path.join('data', 'VLSP2016'),
                 max_length=512,
                 tokenizer_type=BERTvi[0]):
        super(VLSP2016, self).__init__()

        self.df = pd.read_csv(os.path.join(path, file),
                              names=['sentence', 'label'],
                              sep='\t',
                              encoding='utf-8-sig')

        self.max_length = max_length

        self.tokenizer_type = tokenizer_type
        if tokenizer_type == BERTvi[0]:
            self.tokenizer = PhoBertTokenizer(max_length=self.max_length)
        else:
            self.tokenizer = BertViTokenizer(max_length=self.max_length, shortcut_pretrained=BERTvi[1])

        logger.info('Loaded VLSP-2016')
        logger.info(f'There are {len(self.df)} samples in {file} dataset.')

    def __getitem__(self, item):
        text = self.df.iloc[item, 0].encode('utf-8')
        label = self.df.iloc[item, 1]
        text = text.decode('utf-8-sig').strip()

        tent = self.tokenizer(text)
        return tent, 1 if label == 'NEU' else 2 if label == 'NEG' else 0

    def __len__(self):
        return len(self.df)


class AIVIVN(Dataset):
    def __init__(self,
                 file='train.crash',
                 path=os.path.join('data', 'AIVIVN'),
                 max_length=256,
                 tokenizer_type=BERTvi[0],
                 pivot=0.8,
                 train=True,
                 eval=False):
        super(AIVIVN, self).__init__()
        with open(os.path.join(path, file), mode='r', encoding='utf-8-sig') as stream:
            self.train = stream.read()

        if eval is False:
            self.train = self.train.split('\n\ntrain_')
            if train is True:
                self.train = self.train[: int(len(self.train) * pivot)]
            else:
                self.train = self.train[int(len(self.train) * pivot):]
        else:
            self.train = self.train.split('\n\ntest_')

        self.max_length = max_length

        self.tokenizer_type = tokenizer_type
        if tokenizer_type == BERTvi[0]:
            self.tokenizer = PhoBertTokenizer(max_length=self.max_length)
        else:
            self.tokenizer = BertViTokenizer(max_length=self.max_length, shortcut_pretrained=BERTvi[1])

        self.eval = eval

        logger.info('Loaded AIVIVN')
        logger.info(f'There are {len(self.train)} samples in {file} dataset')

    def __getitem__(self, item):
        sample = self.train[item].strip()
        sent = sample[8:-3].strip()
        if self.eval is True:
            sent = sample[8:-1]
            id = sample.split('\n')[0].strip()
            if item != 0:
                id = 'test_' + id
            return id, self.tokenizer(sent)

        label = int(sample[-1:])
        return self.tokenizer(sent), label

    def __len__(self):
        return len(self.train)


class UITVSFC(Dataset):
    def __init__(self,
                 file='train',
                 path=os.path.join('data', 'UIT-VSFC'),
                 max_length=512,
                 tokenizer_type=BERTvi[0]):
        super(UITVSFC, self).__init__()
        with open(os.path.join(path, file, 'sents.txt'), mode='r', encoding='utf-8-sig') as stream:
            self.sents = stream.read().strip().split('\n')
        with open(os.path.join(path, file, 'sentiments.txt'), mode='r', encoding='utf-8-sig') as stream:
            self.sentiments = stream.read().strip().split('\n')
        with open(os.path.join(path, file, 'topics.txt'), mode='r', encoding='utf-8-sig') as stream:
            self.topics = stream.read().strip().split('\n')

        self.max_length = max_length

        self.tokenizer_type = tokenizer_type
        if tokenizer_type == BERTvi[0]:
            self.tokenizer = PhoBertTokenizer(max_length=self.max_length)
        else:
            self.tokenizer = BertViTokenizer(max_length=self.max_length, shortcut_pretrained=BERTvi[1])

        logger.info('Loaded UIT-VSFC')
        logger.info(f'There are {len(self.sents)} samples in {file} dataset.')

    def __getitem__(self, item):
        return self.tokenizer(self.sents[item].strip()), int(self.sentiments[item])

    def __len__(self):
        return len(self.sents)


class VLSP2018(Dataset):
    def __init__(self,
                 data='Hotel',
                 file='train',
                 path=os.path.join('data', 'VLSP2018'),
                 max_length=256,
                 tokenizer_type=BERTvi[0]):
        super(VLSP2018, self).__init__()
        self.max_length = max_length
        with open(os.path.join(path, f'VLSP2018-SA-{data}-{file}.prod'), mode='r', encoding='utf-8-sig') as stream:
            self.file = stream.read()

        self.data = data.lower()

        self.entity_hotel = ['HOTEL', 'ROOMS', 'ROOM_AMENITIES', 'FACILITIES', 'SERVICE', 'LOCATION', 'FOOD&DRINKS']
        self.attribute_hotel = ['GENERAL', 'PRICES', 'DESIGN&FEATURES', 'CLEANLINESS', 'COMFORT', 'QUALITY', 'STYLE&OPTIONS', 'MISCELLANEOUS']
        self.aspect_hotel = [f'{x}#{y}' for x in self.entity_hotel for y in self.attribute_hotel]

        self.entity_restaurant = ['RESTAURANT', 'FOOD', 'DRINKS', 'AMBIENCE', 'SERVICE', 'LOCATION']
        self.attribute_restaurant = ['GENERAL', 'PRICES', 'QUALITY', 'STYLE&OPTIONS', 'MISCELLANEOUS']
        self.aspect_restaurant = [f'{x}#{y}' for x in self.entity_restaurant for y in self.attribute_restaurant]

        self.polarities = ['negative', 'neural', 'positive']

        self.file = self.file.split('\n\n')

        self.tokenizer_type = tokenizer_type
        if tokenizer_type == BERTvi[0]:
            self.tokenizer = PhoBertTokenizer(max_length=self.max_length)
        else:
            self.tokenizer = BertViTokenizer(max_length=self.max_length, shortcut_pretrained=BERTvi[1])

    def label_encode(self, x):
        x = x.split('\n')

        aspect, polarity = x[0].split(',')
        lb = None

        if self.data == 'hotel':
            lb = self.aspect_hotel.index(aspect)
        elif self.data == 'restaurant':
            lb = self.aspect_restaurant.index(aspect)

        polarity = polarity.strip()
        polarity = ['negative', 'neutral', 'positive'].index(polarity)
        aspect = aspect.replace('#', ', ').replace('&', ' and ').lower()
        return aspect, lb, polarity

    def __getitem__(self, item):
        lines = self.file[item].split('\n')
        label = self.label_encode(lines[1].strip())

        if self.data == 'hotel':
            aspect = torch.zeros((self.aspect_hotel.__len__()))
            polarity = torch.zeros((3))
        else:
            aspect = torch.zeros((self.aspect_restaurant.__len__()))
            polarity = torch.zeros((3))

        aspect[label[1]] = 1
        polarity[label[2]] = 1

        text = lines[0].strip()
        return self.tokenizer(text), aspect, polarity

    def __len__(self):
        return self.file.__len__()
