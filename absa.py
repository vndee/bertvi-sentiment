import os
import time
import json
import math
import torch
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from yaml import load
from utils.logger import get_logger
from collections import namedtuple
from utils.loader import VLSP2018
from models.phobert import PhoBertEncoder
from utils.optimizer import set_seed
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from transformers.optimization import AdamW
from transformers import get_cosine_schedule_with_warmup
from classifier.phobert_sequence_clf import PhoBertForSequenceClassification


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
    'multilingual_bert_vlsp_2016.yaml',
    'phobert_vlsp2018.yaml'
]

net = None
arg = argparse.ArgumentParser(description='BERTvi-sentiment Trainer')
arg.add_argument('-f', '--config', default=os.path.join('config', configs[6]))
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


class SelfAttention(torch.nn.Module):
    def __init__(self, num_attention_heads, hidden_size):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(self.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(self.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(self.hidden_size, self.all_head_size)

        self.dense = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)                    # [Batch_size x Num_of_heads x Seq_length x Seq_length]

        context_layer = torch.matmul(attention_probs, value_layer)                # [Batch_size x Num_of_heads x Seq_length x Head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()            # [Batch_size x Seq_length x Num_of_heads x Head_size]

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)              # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer)
        return output[0]


class Model(torch.nn.Module):
    def __init__(self, aspect_num, polarity_num):
        super(Model, self).__init__()
        self.base_encoder = PhoBertEncoder()
        self.lstm_encoder = torch.nn.LSTM(768, 512, bidirectional=True, batch_first=True)
        self.aspect_head = torch.nn.Linear(1024, aspect_num)
        self.polarity_head = torch.nn.Linear(aspect_num + 768, polarity_num)
        self.aspect_one_hot = torch.eye(aspect_num)
        self.aspect_num = aspect_num

    def forward(self, x, attention_mask):
        x = self.base_encoder(x, attention_mask=attention_mask)
        lstm_out, _ = self.lstm_encoder(x[2][-1])
        x = self.aspect_head(lstm_out[:, -1, :])
        x = torch.nn.functional.sigmoid(x)
        mask = (x > 0.5).nonzero()
        ap_var = torch.zeros((x.shape[0], x.shape[1], self.aspect_num))
        # ap_var[mask] = self.aspect_one_hot[mask[:1]]
        print(ap_var)
        return lstm_out


if __name__ == '__main__':
    # config parsing
    opts = config_parsing(args.config)
    set_seed(opts.random_seed)
    logger = get_logger(f'Experiment_{opts.encoder}_{opts.dataset}')
    logger.info(opts)

    dataset = VLSP2018()
    test_dataset = VLSP2018(data='Hotel', file='dev')

    # initialize models
    net = Model(dataset.aspect_hotel.__len__(), 3)
    net = net.to(opts.device)
    print(net)

    if hasattr(opts, 'pretrained'):
        net.load_state_dict(torch.load(opts.pretrained, map_location='cpu' if opts.device == 'cpu' else None)).to(opts.device)
        logger.info(f'Loaded pretrained model {opts.encoder} for {opts.encoder}')

    experiment_path = os.path.join(experiment_path, f'{opts.encoder}_{opts.dataset}')
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        os.makedirs(os.path.join(experiment_path, 'checkpoints'))
        logger.info(f'Create directory {experiment_path}')

    optimizer = AdamW(net.parameters())
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=opts.epochs
                                                   * len(dataset) / opts.batch_size)

    df = pd.DataFrame(columns=['epoch', 'train_acc', 'val_acc', 'train_loss', 'val_loss', 'train_f1', 'val_f1'])
    logger.info('Start training...')
    best_checkpoint = 0.0

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=opts.batch_size,
                                               shuffle=True,
                                               num_workers=opts.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=opts.batch_size,
                                               shuffle=True,
                                               num_workers=opts.num_workers)

    optimizer.zero_grad()
    aspect_criterion = torch.nn.BCELoss()

    for epoch in range(opts.epochs):
        # Training
        net.train()
        _preds, _targets = None, None
        train_loss, train_t = 0.0, 0.0

        steps, total_loss = 0, 0.0
        for idx, (sents, aspect, polarity) in enumerate(tqdm(train_loader, desc=f'Training EPOCH {epoch}/{opts.epochs}')):
            sents, aspect, polarity = torch.tensor(sents, dtype=torch.long, device=opts.device), \
                                      torch.tensor(aspect, dtype=torch.long, device=opts.device), \
                                      torch.tensor(polarity, dtype=torch.long, device=opts.device)

            optimizer.zero_grad()

            mask = (sents > 0).to(opts.device)
            preds = net(sents, attention_mask=mask)
            loss = torch.nn.functional.cross_entropy(preds, aspect)

            loss.backward()
            if idx % opts.accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()

            total_loss = total_loss + loss.item()

            if opts.device == 'cuda':
                preds = preds.detach().cpu().numpy()
                labels = aspect.detach().cpu().numpy()
            else:
                preds = preds.detach().numpy()
                labels = aspect.detach().numpy()

            predicted = np.argmax(preds, 1)

            _preds = np.atleast_1d(predicted) if _preds is None else \
                np.concatenate([_preds, np.atleast_1d(predicted)])
            _targets = np.atleast_1d(labels) if _targets is None else \
                np.concatenate([_targets, np.atleast_1d(labels)])

        train_loss = total_loss / dataset.__len__()
        train_acc, train_f1 = evaluate(_preds, _targets)

        with torch.no_grad():
            # Validation
            net.eval()
            total_loss = 0.0
            _preds, _targets = None, None

            for idx, item in enumerate(tqdm(test_loader, desc=f'Test EPOCH: {epoch}/{opts.epochs}')):
                sents, aspect, polarity = item[0].to(opts.device), \
                                          item[1].to(opts.device), \
                                          item[2].to(opts.device)

                mask = (sents > 0).to(opts.device)
                loss, preds = net(sents, mask)

                total_loss = total_loss + loss.item()

                if opts.device == 'cuda':
                    preds = preds.detach().cpu().numpy()
                    labels = aspect.detach().cpu().numpy()
                else:
                    preds = preds.detach().numpy()
                    labels = aspect.detach().numpy()

                predicted = np.argmax(preds, 1)

                _preds = np.atleast_1d(predicted) if _preds is None else \
                    np.concatenate([_preds, np.atleast_1d(predicted)])
                _targets = np.atleast_1d(labels) if _targets is None else \
                    np.concatenate([_targets, np.atleast_1d(labels)])

            val_loss = total_loss / test_dataset.__len__()
            val_acc, val_f1 = evaluate(_preds, _targets)

            logger.info(f'EPOCH [{epoch}/{opts.epochs}] Training accuracy: {train_acc} | '
                        f'Loss: {train_loss} | '
                        f'F1: {train_f1} | '
                        f'Time: {train_t}s')

            logger.info(f'[EPOCH [{epoch}/{opts.epochs}] Validation accuracy: {val_acc} | '
                        f'Loss: {val_loss} | '
                        f'F1: {val_f1} | '
                        f'Time: {val_acc}s')

            df.loc[len(df)] = [epoch, train_acc, val_acc, train_loss, val_loss, train_f1, val_f1]

            if val_f1 > best_checkpoint:
                best_checkpoint = val_f1
                logger.info(f'New state-of-the-art model detected. Saved to {experiment_path}.')
                torch.save(net.state_dict(), os.path.join(experiment_path, 'checkpoints', f'checkpoint_best.vndee'))
                with open(os.path.join(experiment_path, 'checkpoints', 'best.json'), 'w+') as stream:
                    json.dump({
                        'test_f1': val_f1
                    }, stream)

    # save history to csv
    df.to_csv(os.path.join(experiment_path, 'history.csv'))
    logger.info('Training completed')