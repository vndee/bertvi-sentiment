import os
import time
import json
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from yaml import load
from utils.logger import get_logger
from collections import namedtuple
from utils.loader import VLSP2016, UITVSFC, AIVIVN
from classifier.model import SentimentAnalysisModel
from utils.optimizer import set_seed, create_optimizer
from classifier.phobert_sequence_clf import PhoBertForSequenceClassification
from models.phobert import PhoBertEncoder
from models.bert import BertEncoder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers.optimization import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold, StratifiedKFold


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
arg.add_argument('-f', '--config', default=os.path.join('config', configs[0]))
args = arg.parse_args()


def config_parsing(arg):
    """
    :rtype: Tuple
    """
    data = load(open(arg), Loader=Loader)
    opts = namedtuple('Config', data.keys())(*data.values())
    return opts


def inference(opts, model, inputs, labels, criterion):
    t0 = time.time()

    mask = (inputs > 0).to(opts.device)
    if opts.encoder in ['phobert', 'bert']:
        preds = model(inputs, mask)
        loss = criterion(preds, labels)
    else:
        loss, preds = model(inputs, mask, labels=labels)

    if opts.device == 'cuda':
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    else:
        preds = preds.detach().numpy()
        labels = labels.detach().numpy()

    predicted = np.argmax(preds, 1)
    t1 = time.time()

    return loss, t1 - t0, predicted, labels


			def evaluate(_preds, _targets):
    report = classification_report(_preds,
                                   _targets,
                                   output_dict=True,
                                   zero_division=1)

    acc = report['accuracy']
    f1 = report['macro avg']['f1-score']

    return acc, f1


if __name__ == '__main__':
    # config parsing
    opts = config_parsing(args.config)
    set_seed(opts.random_seed)
    logger = get_logger(f'Experiment_{opts.encoder}_{opts.dataset}')
    logger.info(opts)

    # initialize models
    net = None
    if opts.encoder == 'phobert':
        enc = PhoBertEncoder(frozen=True)
        net = SentimentAnalysisModel(enc, 768, opts.num_classes, device=opts.device).to(opts.device)
    elif opts.encoder == 'bert':
        enc = BertEncoder()
        net = SentimentAnalysisModel(enc, 768, opts.num_classes, device=opts.device).to(opts.device)
    elif opts.encoder == 'roberta_clf':
        net = PhoBertForSequenceClassification()
        net = net.net.to(opts.device)

    net = torch.nn.DataParallel(net)

    if hasattr(opts, 'pretrained'):
        net.load_state_dict(torch.load(opts.pretrained, map_location='cpu' if opts.device == 'cpu' else None)).to(opts.device)
        logger.info(f'Loaded pretrained model {opts.encoder} for {opts.encoder}')

    if opts.dataset == 'vlsp2016':
        dataset = VLSP2016(file='SA-2016.dev',
                           max_length=opts.max_length,
                           tokenizer_type=opts.tokenizer_type)
        test_dataset = VLSP2016(file='SA-2016.dev_test',
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

    # initialize criterion and optimizer
    if opts.encoder in ['phobert', 'bert']:
        criterion = nn.CrossEntropyLoss()
        optimizer, lr_scheduler, constant_scheduler = create_optimizer(net, opts, len(dataset))
    else:
        optimizer = AdamW(net.parameters())
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=opts.epochs
                                                       * len(dataset) / opts.batch_size)

    df = pd.DataFrame(columns=['fold', 'epoch', 'train_acc', 'val_acc', 'test_acc', 'train_loss', 'val_loss',
                               'test_loss', 'train_f1', 'val_f1', 'test_f1', 'train_t', 'val_t', 'test_t'])
    logger.info('Start training...')
    best_checkpoint = 0.0

    # K-Fold Splitting
    X, y = list(), list()
    for it, lb in dataset:
        X.append(it)
        y.append(lb)

    X, y = torch.stack(X), torch.tensor(y, dtype=torch.long)
    folds = list(StratifiedKFold(n_splits=opts.kfold, shuffle=True, random_state=opts.random_seed).split(X, y))
    test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    for fold, (train_idx, val_idx) in enumerate(folds):
        logger.info(f'Training for fold number {fold}.')
        train_fold = torch.utils.data.TensorDataset(X[train_idx], y[train_idx])
        valid_fold = torch.utils.data.TensorDataset(X[val_idx], y[val_idx])

        train_loader = torch.utils.data.DataLoader(train_fold,
                                                   batch_size=opts.batch_size,
                                                   shuffle=True,
                                                   num_workers=opts.num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_fold,
                                                   batch_size=opts.batch_size,
                                                   shuffle=True,
                                                   num_workers=opts.num_workers)

        optimizer.zero_grad()

        for epoch in range(opts.epochs):
            # Training
            net.train()
            _preds, _targets = None, None
            train_loss, train_t = 0.0, 0.0

            for idx, item in enumerate(tqdm(train_loader, desc=f'[F{fold}] Training EPOCH {epoch}/{opts.epochs}')):
                sents, labels = item[0].to(opts.device), \
                                item[1].to(opts.device)

                optimizer.zero_grad()
                _loss, _t, predicted, labels = inference(opts, net, sents, labels, criterion)

                _loss.backward()

                if idx % opts.accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()

                _preds = np.atleast_1d(predicted) if _preds is None else \
                    np.concatenate([_preds, np.atleast_1d(predicted)])
                _targets = np.atleast_1d(labels) if _targets is None else \
                    np.concatenate([_targets, np.atleast_1d(labels)])

                train_loss = train_loss + _loss.item()
                train_t = train_t + _t

            train_loss = train_loss / train_fold.__len__()
            train_acc, train_f1 = evaluate(_preds, _targets)

            with torch.no_grad():
                # Validation
                net.eval()
                val_loss, val_t = 0.0, 0.0
                _preds, _targets = None, None

                for idx, item in enumerate(tqdm(valid_loader,
                                                desc=f'[F{fold}] Validation EPOCH: {epoch}/{opts.epochs}')):
                    sents, labels = item[0].to(opts.device), \
                                    item[1].to(opts.device)

                    _loss, _t, predicted, labels = inference(opts, net, sents, labels, criterion)

                    _preds = np.atleast_1d(predicted) if _preds is None else \
                        np.concatenate([_preds, np.atleast_1d(predicted)])
                    _targets = np.atleast_1d(labels) if _targets is None else \
                        np.concatenate([_targets, np.atleast_1d(labels)])

                    val_loss = val_loss + _loss.item()
                    val_t = val_t + _t

                val_loss = val_loss / valid_fold.__len__()
                val_acc, val_f1 = evaluate(_preds, _targets)

                # Testing
                test_loss, test_t = 0.0, 0.0
                _preds, _targets = None, None
                for idx, item in enumerate(tqdm(test_loader, desc=f'[F{fold}] Test EPOCH: {epoch}/{opts.epochs}')):
                    sents, labels = item[0].to(opts.device), \
                                    item[1].to(opts.device)

                    _loss, _t, predicted, labels = inference(opts, net, sents, labels, criterion)

                    _preds = np.atleast_1d(predicted) if _preds is None else \
                        np.concatenate([_preds, np.atleast_1d(predicted)])
                    _targets = np.atleast_1d(labels) if _targets is None else \
                        np.concatenate([_targets, np.atleast_1d(labels)])

                    test_loss = test_loss + _loss.item()
                    test_t = test_t + _t

                test_loss = test_loss / test_dataset.__len__()
                test_acc, test_f1 = evaluate(_preds, _targets)

                logger.info(f'[F{fold}] EPOCH [{epoch}/{opts.epochs}] Training accuracy: {train_acc} | '
                            f'Loss: {train_loss} | '
                            f'F1: {train_f1} | '
                            f'Time: {train_t}s')

                logger.info(f'[F{fold}] EPOCH [{epoch}/{opts.epochs}] Validation accuracy: {val_acc} | '
                            f'Loss: {val_loss} | '
                            f'F1: {val_f1} | '
                            f'Time: {val_t}s')

                logger.info(f'[F{fold}] EPOCH [{epoch}/{opts.epochs}] Test accuracy: {test_acc} | '
                            f'Loss: {test_loss} | '
                            f'F1: {test_f1} | '
                            f'Time: {test_t}s')

                df.loc[len(df)] = [fold, epoch, train_acc, val_acc, test_acc, train_loss, val_loss, test_loss,
                                   train_f1, val_f1, test_f1, train_t, val_t, test_t]

                if test_f1 > best_checkpoint:
                    best_checkpoint = test_f1
                    logger.info(f'[F{fold}] New state-of-the-art model detected. Saved to {experiment_path}.')
                    torch.save(net.state_dict(), os.path.join(experiment_path, 'checkpoints', f'checkpoint_best.vndee'))
                    with open(os.path.join(experiment_path, 'checkpoints', 'best.json'), 'w+') as stream:
                        json.dump({
                            'test_f1': test_f1
                        }, stream)

    # save history to csv
    df.to_csv(os.path.join(experiment_path, 'history.csv'))
    logger.info('Training completed')
