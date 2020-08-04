import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from loader import get_logger
from train import config_parsing
from loader import VLSP2016, AIVIVN, UITVSFC


experiment_path = 'outputs'


if __name__ == '__main__':
    arg = argparse.ArgumentParser('Sentiment Analysis Evaluation Module')
    arg.add_argument('-f', '--config', default=os.path.join('config', 'phobert_aivivn.yaml'))
    args = arg.parse_args()
    args = config_parsing(args.config)
    logger = get_logger(f'Evaluation_{args.encoder}_{args.dataset}')

    net = torch.load(args.pretrained)
    net.to(args.device)
    net.eval()

    logger.info(f'Loaded {args.encoder}-{args.dataset}')

    if args.dataset == 'aivivn':
        dataset = AIVIVN(file='test.crash',
                         tokenizer_type=args.tokenizer_type,
                         eval=True,
                         max_length=args.max_length)
    else:
        dataset = []

    df = pd.DataFrame(columns=['id', 'label'])

    for id, sents in tqdm(dataset):
        sents = sents.unsqueeze(0)

        with torch.no_grad():
            try:
                preds = net(sents.to(args.device), (sents > 0).to(args.device))
                preds = F.softmax(preds, dim=1)
                if args.device == 'cuda':
                    preds = preds.detach().cpu().numpy()
                preds = np.argmax(preds, 1)[0]
                df.append([id, preds])
            except Exception as ex:
                logger.info(sents)
                logger.exception(ex)

    print(df.head())
    df.to_csv(os.path.join(experiment_path, 'submission.csv'))
    logger.info('Test completed.')