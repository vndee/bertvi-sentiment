import os
import torch
import datetime
import torch.nn as nn
from collections import namedtuple
from logger import get_logger
from yaml import load
from model import SentimentAnalysisModel
from phobert import PhoBertEncoder
from loader import VLSP2016
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except Exception as ex:
    from yaml import Loader, Dumper

logger = get_logger('Trainer')
# writer = SummaryWriter(os.path.join('runs', f'nics2020_{datetime.datetime.now()}'))

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


def diff(x, y):
    correct, incorrect = 0, 0
    for i, j in zip(x, y):
        if i == j:
            correct = correct + 1
        else:
            incorrect = incorrect + 1

    return correct, incorrect


if __name__ == '__main__':
    opts = config_parsing(config)

    enc = PhoBertEncoder()
    net = SentimentAnalysisModel(enc, 768, 2).to(opts.device)
    # writer.add_graph(net)

    dataset = VLSP2016()
    test_dataset = VLSP2016(file='SA-2016.test')
    data_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=0.001,
                                momentum=0.9)

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
            correct = (predicted == labels).sum().item()

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            logger.info(f'[{idx}/{len(data_loader)}] Training loss: {loss.item()}')

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
                correct = (predicted == labels).sum().item()

            logger.info(f'EPOCH [{epoch}/{opts.epochs}] Evaluation accuracy: {correct/total}')
