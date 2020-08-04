import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from yaml import load
from optimizer import set_seed, create_optimizer
import matplotlib.pyplot as plt
from loader import VLSP2016, UITVSFC, AIVIVN
from logger import get_logger
from collections import namedtuple
from model import SentimentAnalysisModel
from phobert import PhoBertEncoder
from bert import BertEncoder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except Exception as ex:
    from yaml import Loader, Dumper

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
arg.add_argument('-f', '--config', default=os.path.join('config', configs[3]))
args = arg.parse_args()


def config_parsing(arg):
    data = load(open(arg), Loader=Loader)
    opts = namedtuple('Config', data.keys())(*data.values())
    return opts


if __name__ == '__main__':
    # config parsing
    opts = config_parsing(args.config)
    set_seed(opts.random_seed)
    logger = get_logger(f'Experiment_{opts.encoder}_{opts.dataset}')
    logger.info(opts)

    # initialize model
    net = None
    if opts.encoder == 'phobert':
        enc = PhoBertEncoder()
        net = SentimentAnalysisModel(enc, 768, opts.num_classes).to(opts.device)
    elif opts.encoder == 'bert':
        enc = BertEncoder()
        net = SentimentAnalysisModel(enc, 768, opts.num_classes).to(opts.device)

    if hasattr(opts, 'pretrained'):
        net = torch.load(opts.pretrained)
        logger.info(f'Loaded pretrained model {opts.encoder} for {opts.encoder}')

    if opts.dataset == 'vlsp2016':
        dataset = VLSP2016(file='SA-2016.train',
                           max_length=opts.max_length,
                           tokenizer_type=opts.tokenizer_type)
        test_dataset = VLSP2016(file='SA-2016.test',
                                max_length=opts.max_length,
                                tokenizer_type=opts.tokenizer_type)
    elif opts.dataset == 'uit-vsfc':
        dataset = UITVSFC(file='train',
                          max_length=opts.max_length,
                          tokenizer_type=opts.tokenizer_type)
        test_dataset = UITVSFC(file='test',
                               max_length=opts.max_length,
                               tokenizer_type=opts.tokenizer_type)
    else:
        dataset = AIVIVN(file='dev.crash',
                         max_length=opts.max_length,
                         tokenizer_type=opts.tokenizer_type,
                         pivot=opts.pivot,
                         train=True)
        test_dataset = AIVIVN(file='dev.crash',
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
    optimizer, linear_scheduler, constant_scheduler = create_optimizer(net, opts, len(dataset))

    df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'f1_neg', 'train_time', 'val_time'])
    logger.info('Start training...')
    best_checkpoint = 0.0

    # train
    for epoch in range(opts.epochs):
        t0 = time.time()
        epoch = epoch + 1
        total = 0
        total_loss = 0.0
        train_preds = None
        train_targets = None

        net.train()
        for idx, item in enumerate(tqdm(data_loader, desc=f'Training EPOCH {epoch}/{opts.epochs}')):
        # for idx, item in enumerate(data_loader):
            sents, labels = item[0].to(opts.device), \
                            item[1].to(opts.device)

            optimizer.zero_grad()
            try:
                preds = net(sents, (sents > 0).to(opts.device))
            except Exception as ex:
                logger.exception(ex)
                continue

            loss = criterion(preds, labels)

            if opts.device == 'cuda':
                preds = preds.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            else:
                preds = preds.detach().numpy()
                labels = labels.detach().numpy()

            predicted = np.argmax(preds, 1)
            total += labels.shape[0]

            train_preds = np.atleast_1d(predicted) if train_preds is None else \
                np.concatenate([train_preds, np.atleast_1d(predicted)])
            train_targets = np.atleast_1d(labels) if train_targets is None else \
                np.concatenate([train_targets, np.atleast_1d(labels)])

            loss.backward()
            if idx % opts.accumulation_steps == 0:
                optimizer.step()
                linear_scheduler.step()

            total_loss = total_loss + loss.item()
            # logger.info(f'[{idx + 1}/{len(data_loader)}] Training loss: {loss.item()}')

        train_loss = float(total_loss / total)

        report = classification_report(train_preds,
                                       train_targets,
                                       output_dict=True,
                                       zero_division=1)

        t1 = time.time()
        train_acc = report['accuracy']
        train_neg_f1 = report['0']['f1-score']

        with torch.no_grad():
            total = 0
            val_loss = 0.0
            val_preds = None
            val_targets = None

            net.eval()
            for idx, item in enumerate(tqdm(test_data_loader, desc=f'Validation EPOCH: {epoch}/{opts.epochs}')):
                sents, labels = item[0].to(opts.device), \
                                item[1].to(opts.device)

                try:
                    preds = net(sents, (sents > 0).to(opts.device))
                except Exception as ex:
                    logger.exception(ex)
                    continue

                loss = criterion(preds, labels)

                if opts.device == 'cuda':
                    preds = preds.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                else:
                    preds = preds.detach().numpy()
                    labels = labels.detach().numpy()

                predicted = np.argmax(preds, 1)
                total += labels.shape[0]

                val_preds = np.atleast_1d(predicted) if val_preds is None else \
                    np.concatenate([val_preds, np.atleast_1d(predicted)])
                val_targets = np.atleast_1d(labels) if val_targets is None else \
                    np.concatenate([val_targets, np.atleast_1d(labels)])

                val_loss += loss.item()

            val_loss = float(val_loss / total)

            report = classification_report(val_preds,
                                           val_targets,
                                           output_dict=True,
                                           zero_division=1)

            t2 = time.time()
            val_acc = report['accuracy']
            neg_f1 = report['0']['f1-score']

            logger.info(f'EPOCH [{epoch}/{opts.epochs}] Training accuracy: {train_acc} | '
                        f'Training loss: {train_loss} | '
                        f'Negative F1: {train_neg_f1} | '
                        f'Training time: {t1 - t0}s')

            logger.info(f'EPOCH [{epoch}/{opts.epochs}] Validation accuracy: {val_acc} | '
                        f'Validation loss: {train_loss} | '
                        f'Negative F1: {neg_f1} | '
                        f'Validation time: {t2 - t1}s')

            df.loc[len(df)] = [epoch, train_loss, train_acc, val_loss, val_acc, neg_f1, t1 - t0, t2 - t1]

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
