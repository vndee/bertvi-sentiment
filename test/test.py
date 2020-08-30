import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from utils.loader import get_logger
from train import config_parsing
from utils.loader import AIVIVN
from models.phobert import PhoBertEncoder
from classifier.model import SentimentAnalysisModel

experiment_path = 'outputs'


if __name__ == '__main__':
    arg = argparse.ArgumentParser('Sentiment Analysis Evaluation Module')
    arg.add_argument('-f', '--config', default=os.path.join('config', 'phobert_aivivn.yaml'))
    args = arg.parse_args()
    args = config_parsing(args.config)
    logger = get_logger(f'Evaluation_{args.encoder}_{args.dataset}')
    logger.info(args)

    enc = PhoBertEncoder()
    net = SentimentAnalysisModel(enc, 768, args.num_classes, device=args.device)
    net = torch.nn.DataParallel(net)

    net.eval()

    if hasattr(args, 'pretrained'):
        net.load_state_dict(torch.load(args.pretrained))
        logger.info(f'Loaded pretrained model {args.encoder} for {args.encoder}')

    logger.info(f'Loaded {args.encoder}-{args.dataset}')

    if args.dataset == 'aivivn':
        dataset = AIVIVN(file='test.crash',
                         tokenizer_type=args.tokenizer_type,
                         eval=True,
                         max_length=args.max_length)
    else:
        dataset = []

    df = pd.DataFrame(columns=['id', 'label'])

    for idx, (id, sents) in enumerate(tqdm(dataset)):
        sents = sents.unsqueeze(0)

        with torch.no_grad():
            try:
                preds = net(sents.to(args.device), (sents > 0).to(args.device))
                preds = F.softmax(preds, dim=1)
                if args.device == 'cuda':
                    preds = preds.detach().cpu().numpy()
                preds = np.argmax(preds, 1)[0]
                df.loc[len(df)] = [id, preds]
            except Exception as ex:
                logger.info(sents)
                logger.exception(ex)

    print(df.head())
    df.to_csv(os.path.join(experiment_path, 'submission.csv'), index=False, columns=['id', 'label'])
    logger.info('Test completed.')