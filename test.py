import os
import torch
import argparse
from loader import get_logger
from train import config_parsing
from loader import VLSP2016, AIVIVN, UITVSFC


if __name__ == '__main__':
    arg = argparse.ArgumentParser('Sentiment Analysis Evaluation Module')
    arg.add_argument('-f', '--config', default=os.path.join('config', 'phobert_aivivn.yaml'))
    args = arg.parse_args()
    args = config_parsing(args.config)
    logger = get_logger(f'Evaluation_{args.encoder}_{args.dataset}')

    net = torch.load(args.pretrained)
    net.to(args.device)
    net.eval()

    logger.info(net)
