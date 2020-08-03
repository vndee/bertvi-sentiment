import os
import torch
import torch.nn as nn
from yaml import load
from loader import VLSP2016, UITVSFC, AIVIVN
from logger import get_logger
from collections import namedtuple
from model import SentimentAnalysisModel
from phobert import PhoBertEncoder
from bert import BertEncoder
from torch.utils.data import DataLoader

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except Exception as ex:
    from yaml import Loader, Dumper

logger = get_logger('Trainer')

configs = [
    'phobert_vlsp_2016.yaml',
    'phobert_uit_vsfc.yaml',
    'phobert_aivivn.yaml',
    'multilingual_bert_aivivn.yaml',
    'multilingual_bert_uit_vsfc.yaml',
    'multilingual_bert_vlsp_2016.yaml'
]

config = os.path.join('config', configs[0])


def config_parsing(arg):
    data = load(open(arg), Loader=Loader)
    opts = namedtuple('Config', data.keys())(*data.values())
    return opts


if __name__ == '__main__':
    opts = config_parsing(config)
    logger.info(opts)

    if opts.encoder == 'phobert':
        enc = PhoBertEncoder()
        net = SentimentAnalysisModel(enc, 768, opts.num_classes).to(opts.device)
    elif opts.encoder == 'bert':
        enc = BertEncoder()
        net = SentimentAnalysisModel(enc, 768, opts.num_classes).to(opts.device)

    if opts.dataset == 'vlsp2016':
        dataset = VLSP2016(file='SA-2016.dev',
                           max_length=opts.max_length,
                           tokenizer_type=opts.tokenizer_type)
        test_dataset = VLSP2016(file='SA-2016.dev',
                                max_length=opts.max_length,
                                tokenizer_type=opts.tokenizer_type)
    elif opts.dataset == 'uit-vsfc':
        dataset = UITVSFC(file='dev',
                          max_length=opts.max_length,
                          tokenizer_type=opts.tokenizer_type)
        test_dataset = UITVSFC(file='dev',
                               max_length=opts.max_length,
                               tokenizer_type=opts.tokenzier_type)
    else:
        dataset = AIVIVN(file='train.crash',
                         max_length=opts.max_length,
                         tokenizer_type=opts.tokenizer_type,
                         pivot=opts.pivot,
                         train=True)
        test_dataset = AIVIVN(file='train.crash',
                              max_length=opts.max_length,
                              tokenizer_type=opts.tokenizer_type,
                              pivot=opts.pivot,
                              train=False)

    data_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=opts.learning_rate,
                                momentum=opts.momentum)

    logger.info('Start training...')
    for epoch in range(opts.epochs):
        correct, total = 0, 0
        for idx, item in enumerate(data_loader):
            sents, labels = item[0].to(opts.device), \
                            item[1].to(opts.device)

            optimizer.zero_grad()
            preds = net(sents)

            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            logger.info(f'[{idx}/{len(data_loader)}] Training loss: {loss.item()}')

        logger.info(f'correct: {correct} | total: {total}')
        logger.info(f'EPOCH [{epoch}/{opts.epochs}] Training accuracy: {correct/total}')

        with torch.no_grad():
            correct, total = 0, 0
            for idx, item in enumerate(test_data_loader):
                sents, labels = item[0].to(opts.device), \
                                item[1].to(opts.device)

                preds = net(sents)
                loss = criterion(preds, labels)

                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            logger.info(f'EPOCH [{epoch}/{opts.epochs}] Evaluation accuracy: {correct/total}')
