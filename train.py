import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from yaml import load
import matplotlib.pyplot as plt
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
experiment_path = 'outputs'

configs = [
    'phobert_vlsp_2016.yaml',
    'phobert_uit_vsfc.yaml',
    'phobert_aivivn.yaml',
    'multilingual_bert_aivivn.yaml',
    'multilingual_bert_uit_vsfc.yaml',
    'multilingual_bert_vlsp_2016.yaml'
]

arg = argparse.ArgumentParser(description='BERTvi-sentiment Trainer')
arg.add_argument('-f', '--config', default=os.path.join('config', configs[0]))
args = arg.parse_args()


def config_parsing(arg):
    data = load(open(arg), Loader=Loader)
    opts = namedtuple('Config', data.keys())(*data.values())
    return opts


if __name__ == '__main__':
    # config parsing
    opts = config_parsing(args.config)
    logger.info(opts)

    # initialize model
    if opts.encoder == 'phobert':
        enc = PhoBertEncoder()
        net = SentimentAnalysisModel(enc, 768, opts.num_classes).to(opts.device)
    elif opts.encoder == 'bert':
        enc = BertEncoder()
        net = SentimentAnalysisModel(enc, 768, opts.num_classes).to(opts.device)

    if opts.dataset == 'vlsp2016':
        dataset = VLSP2016(file='SA-2016.train',
                           max_length=opts.max_length,
                           tokenizer_type=opts.tokenizer_type)
        test_dataset = VLSP2016(file='SA-2016.train',
                                max_length=opts.max_length,
                                tokenizer_type=opts.tokenizer_type)
    elif opts.dataset == 'uit-vsfc':
        dataset = UITVSFC(file='train',
                          max_length=opts.max_length,
                          tokenizer_type=opts.tokenizer_type)
        test_dataset = UITVSFC(file='test',
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

    experiment_path = os.path.join(experiment_path, f'{opts.encoder}_{opts.dataset}')
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        os.makedirs(os.path.join(experiment_path, 'checkpoints'))
        logger.info(f'Create directory {experiment_path}')

    # load data
    data_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers,
                             drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers,
                                  drop_last=True)

    # initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=opts.learning_rate,
                                momentum=opts.momentum)

    df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'train_time', 'val_time'])
    logger.info('Start training...')
    best_checkpoint = 0.0

    # train
    for epoch in range(opts.epochs):
        t0 = time.time()
        epoch = epoch + 1
        correct, total = 0, 0
        total_loss = 0.0
        for idx, item in enumerate(data_loader):
            sents, labels = item[0].to(opts.device), \
                            item[1].to(opts.device)

            optimizer.zero_grad()
            try:
                preds = net(sents)
            except Exception as ex:
                logger.exception(ex)
                logger.info(sents)
                logger.info(labels)
                break

            loss = criterion(preds, labels)

            if opts.device == 'cuda':
                preds = preds.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            else:
                preds = preds.detach().numpy()
                labels = labels.detach().numpy()

            predicted = np.argmax(preds, 1)
            total += labels.shape[0]
            correct += np.sum((predicted == labels))

            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
            logger.info(f'[{idx + 1}/{len(data_loader)}] Training loss: {loss.item()}')

        train_loss = float(total_loss / total)
        train_acc = float(correct / total)

        t1 = time.time()
        logger.info(f'EPOCH [{epoch}/{opts.epochs}] Training accuracy: {train_acc} | '
                    f'Training loss: {train_loss} | '
                    f'Training time: {t1 - t0}s')

        with torch.no_grad():
            correct, total = 0, 0
            val_loss = 0.0
            for idx, item in enumerate(tqdm(test_data_loader)):
                sents, labels = item[0].to(opts.device), \
                                item[1].to(opts.device)

                try:
                    preds = net(sents)
                except Exception as ex:
                    logger.exception(ex)
                    logger.info(sents)
                    logger.info(labels)
                    break

                loss = criterion(preds, labels)

                if opts.device == 'cuda':
                    preds = preds.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                else:
                    preds = preds.detach().numpy()
                    labels = labels.detach().numpy()

                predicted = np.argmax(preds, 1)
                total += labels.shape[0]
                correct += np.sum((predicted == labels))

                val_loss += loss.item()

            val_loss = float(val_loss / total)
            val_acc = float(correct / total)

            t2 = time.time()
            logger.info(f'EPOCH [{epoch}/{opts.epochs}] Validation accuracy: {val_acc} | '
                        f'Validation loss: {train_loss} | '
                        f'Validation time: {t2 - t1}s')
            df.loc[len(df)] = [epoch, train_loss, train_acc, val_loss, val_acc, t1 - t0, t2 - t1]

            if val_acc > best_checkpoint:
                logger.info(f'New state-of-the-art model detected. Saved to {experiment_path}.')
                torch.save(net, os.path.join(experiment_path, 'checkpoints', f'checkpoint_best.vndee'))

    # save history to csv
    df.to_csv(os.path.join(experiment_path, 'history.csv'))

    # plot figure
    labels = ['Train loss', 'Validation loss', 'Train accuracy', 'Validation accuracy']
    fig, ax1 = plt.subplots()
    # plt.xticks(df['epoch'].astype(int).tolist())

    ax1.set_xlabel('epoch(s)')
    ax1.set_ylabel('loss')

    l1 = ax1.plot(df['epoch'].astype(int).tolist(), df['train_loss'].tolist(), 'b', label=labels[0])[0]
    l2 = ax1.plot(df['epoch'].astype(int).tolist(), df['val_loss'].tolist(), 'g', label=labels[1])[0]

    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')

    l3 = ax2.plot(df['epoch'].astype(int).tolist(), df['train_acc'].tolist(), 'r', label=labels[2])[0]
    l4 = ax2.plot(df['epoch'].astype(int).tolist(), df['val_acc'].tolist(), 'y', label=labels[3])[0]

    fig.legend([l1, l2, l3, l4],
               labels,
               bbox_to_anchor=(0.45, 0.35))

    fig.tight_layout()
    plt.savefig(os.path.join(experiment_path, f'{opts.encoder}_{opts.dataset}.png'),
                dpi=500)
    plt.savefig(os.path.join(experiment_path, f'{opts.encoder}_{opts.dataset}.pdf'),
                dpi=500,
                format='pdf')
    plt.show()
