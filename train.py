import os
import time
import json
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn

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

net = None
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


def inference(opts, inputs, labels, criterion):
    t0 = time.time()

    mask = (inputs > 0).to(opts.device)
    if opts.encoder in ['phobert', 'bert']:
        preds = net(inputs, mask)
        loss = criterion(preds, labels)
    else:
        loss, preds = net(inputs, mask, labels=labels)

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
    if opts.encoder == 'phobert':
        enc = PhoBertEncoder()
        net = SentimentAnalysisModel(enc, 768, opts.num_classes, device=opts.device).to(opts.device)
    elif opts.encoder == 'bert':
        enc = BertEncoder()
        net = SentimentAnalysisModel(enc, 768, opts.num_classes, device=opts.device).to(opts.device)
    elif opts.encoder == 'roberta_clf':
        net = PhoBertForSequenceClassification()
        net = net.net.to(opts.device)

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
        optimizer, lr_scheduler, constant_scheduler = create_optimizer(net, opts, len(dataset))
    else:
        optimizer = AdamW(net.parameters())
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=opts.epochs
                                                       * len(dataset) / opts.batch_size)

    df = pd.DataFrame(columns=['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss', 'train_f1', 'val_f1'])

    logger.info('Start training...')
    best_checkpoint = 0.0

    logger.info(f'Training..')
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=opts.batch_size,
                                               shuffle=True,
                                               num_workers=opts.num_workers)
    valid_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=opts.batch_size,
                                               shuffle=True,
                                               num_workers=opts.num_workers)

    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(opts.epochs):
        # Training
        net.train()
        _preds, _targets = None, None
        train_loss = 0.0

        step, loss = 0, None
        for idx, item in enumerate(tqdm(train_loader, desc=f'Training EPOCH {epoch}/{opts.epochs}')):
            sents, labels = item[0].to(opts.device), \
                            item[1].to(opts.device)

            optimizer.zero_grad()
            mask = (sents > 0).to(opts.device)
            preds = net(sents, attention_mask=mask)
            loss = criterion(preds, labels)

            loss.backward()
            if idx % opts.accumulation_steps:
                optimizer.step()
                lr_scheduler.step()

            if opts.device == 'cuda':
                preds = preds.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            else:
                preds = preds.detach().numpy()
                labels = labels.detach().numpy()

            predicted = np.argmax(preds, 1)
            _preds = np.atleast_1d(predicted) if _preds is None else \
                np.concatenate([_preds, np.atleast_1d(predicted)])
            _targets = np.atleast_1d(labels) if _targets is None else \
                np.concatenate([_targets, np.atleast_1d(labels)])

            train_loss = train_loss + loss.item()

        train_loss = train_loss / dataset.__len__()
        train_acc, train_f1 = evaluate(_preds, _targets)

        with torch.no_grad():
            # Validation
            net.eval()
            val_loss = 0.0
            _preds, _targets = None, None

            for idx, item in enumerate(tqdm(valid_loader,
                                            desc=f'Validation EPOCH: {epoch}/{opts.epochs}')):
                sents, labels = item[0].to(opts.device), \
                                item[1].to(opts.device)

                mask = (sents > 0).to(opts.device)
                preds = net(sents, attention_mask=mask)
                loss = criterion(preds, labels)

                if opts.device == 'cuda':
                    preds = preds.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                else:
                    preds = preds.detach().numpy()
                    labels = labels.detach().numpy()

                predicted = np.argmax(preds, 1)
                _preds = np.atleast_1d(predicted) if _preds is None else \
                    np.concatenate([_preds, np.atleast_1d(predicted)])
                _targets = np.atleast_1d(labels) if _targets is None else \
                    np.concatenate([_targets, np.atleast_1d(labels)])

                val_loss = val_loss + loss.item()

            val_loss = val_loss / test_dataset.__len__()
            val_acc, val_f1 = evaluate(_preds, _targets)

            logger.info(f'EPOCH [{epoch}/{opts.epochs}] Training accuracy: {train_acc} | '
                        f'Loss: {train_loss} | '
                        f'F1: {train_f1}')

            logger.info(f'EPOCH [{epoch}/{opts.epochs}] Validation accuracy: {val_acc} | '
                        f'Loss: {val_loss} | '
                        f'F1: {val_f1}')

            df.loc[len(df)] = [epoch, train_acc, val_acc, train_loss, val_loss, train_f1, val_f1]

            if val_f1 > best_checkpoint:
                best_checkpoint = val_loss
                logger.info(f'New state-of-the-art model detected. Saved to {experiment_path}.')
                torch.save(net.state_dict(), os.path.join(experiment_path, 'checkpoints', f'checkpoint_best.vndee'))
                with open(os.path.join(experiment_path, 'checkpoints', 'best.json'), 'w+') as stream:
                    json.dump({
                        'test_f1': val_f1
                    }, stream)

    # save history to csv
    df.to_csv(os.path.join(experiment_path, 'history.csv'))
    logger.info('Training completed')