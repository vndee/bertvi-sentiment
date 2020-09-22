import os
import torch
import argparse
import pandas as pd
from utils.logger import get_logger

from transformers import AdamW
from transformers import Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = get_logger('Transformers for VSA')


def load_data(root='data', data='VLSP2016', is_train=True):
    if data == 'VLSP2016':
        df = pd.read_csv(os.path.join(root, data, 'SA-2016.train' if is_train else 'SA-2016.test'),
                         names=['sentence', 'label'],
                         sep='\t',
                         encoding='utf-8-sig')

        texts, labels = df['sentence'].apply(lambda row: row.strip()).tolist(), \
                        df['label'].apply(lambda row: 0 if row == 'NEG' else 1 if row == 'NEU' else 2).tolist()
    elif data == 'UIT-VSFC':
        sentences = open(os.path.join(root, data, 'train' if is_train else 'test', 'sents.txt'),
                         mode='r',
                         encoding='utf-8-sig').read().strip().split('\n')
        sentiment = open(os.path.join(root, data, 'train' if is_train else 'test', 'sentiments.txt'),
                         mode='r',
                         encoding='utf-8-sig').read().strip().split('\n')
        texts, labels = [text.strip() for text in sentences], [int(label) for label in sentiment]
    else:
        pivot = 0.9
        file_reader = open(os.path.join(root, data, 'train.crash'), mode='r', encoding='utf-8-sig').read().strip()
        file_reader = file_reader.split('\n\ntrain_')
        texts, labels = [sent[8: -3].strip() for sent in file_reader], [int(sent[-1:]) for sent in file_reader]
        texts, labels = texts[: int(len(texts) * pivot)] if is_train else texts[int(len(texts) * pivot):], \
                        labels[: int(len(labels) * pivot)] if is_train else labels[int(len(labels) * pivot):]
    return texts, labels


class SentimentAnalysisDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

        assert hasattr(self.encodings, 'input_ids'), TypeError('There is no input_ids in sentence encoding.')
        assert hasattr(self.encodings, 'attention_mask'), TypeError('There is no attention_mask in sentence encoding.')
        assert hasattr(self.encodings, 'token_type_ids'), TypeError('There is no token_type_ids in sentence encoding.')
        assert self.encodings['input_ids'].__len__() == self.labels.__len__(), IndexError

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.labels.__len__()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


if __name__ == '__main__':
    # argument parsing
    argument_parser = argparse.ArgumentParser(description='Fine-tune transformer models for '
                                                          'Vietnamese Sentiment Analysis.')
    argument_parser.add_argument('--model', type=str, default='bert')
    argument_parser.add_argument('--freeze_encoder', type=bool, default=True)
    argument_parser.add_argument('--epoch', type=int, default=1)
    argument_parser.add_argument('--learning_rate', type=int, default=1e-5)
    argument_parser.add_argument('--accumulation_step', type=int, default=50)
    argument_parser.add_argument('--device', type=str, default='cuda')
    argument_parser.add_argument('--root', type=str, default='data')
    argument_parser.add_argument('--data', type=str, default='VLSP2016')
    argument_parser.add_argument('--batch_size', type=int, default=1)
    argument_parser.add_argument('--max_length', type=int, default=256)
    argument_parser.add_argument('--num_labels', type=int, default=3)
    argument_parser.add_argument('--warmup_steps', type=int, default=100)
    argument_parser.add_argument('--weight_decay', type=float, default=0.01)
    argument_parser.add_argument('--save_steps', type=int, default=10)
    argument_parser.add_argument('--logging_steps', type=int, default=10)
    args = argument_parser.parse_args()
    logger.info(args)

    # load sentiment data
    train_texts, train_labels = load_data(root=args.root, data=args.data, is_train=True)
    test_texts, test_labels = load_data(root=args.root, data=args.data, is_train=False)
    assert train_texts.__len__() == train_labels.__len__(), IndexError
    assert test_texts.__len__() == test_labels.__len__(), IndexError

    # init model
    if args.model == 'bert':
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.num_labels = args.num_labels

        net = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # encode data
    train_encodings, test_encodings = tokenizer(train_texts,
                                                truncation=True,
                                                padding=True,
                                                max_length=args.max_length),\
                                      tokenizer(test_texts,
                                                truncation=True,
                                                padding=True,
                                                max_length=args.max_length)
    train_dataset, test_dataset = SentimentAnalysisDataset(train_encodings, train_labels), \
                                  SentimentAnalysisDataset(test_encodings, test_labels)

    # freeze encoder
    if args.freeze_encoder is True:
        for param in net.base_model.parameters():
            param.requires_grad = True
    logger.info(f'Model Architecture: {net}')

    # init optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
    logger.info(f'Optimizer {optimizer}')

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluate_during_training=True,
        logging_dir='./logs',
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        no_cuda=False
    )

    trainer = Trainer(
        model=net,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    train_output = trainer.train()
    evaluate_output = trainer.evaluate(eval_dataset=test_dataset)

    logger.info(train_output)
    logger.info(evaluate_output)
