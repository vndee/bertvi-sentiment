import os
import torch
import numpy as np
from tqdm import tqdm
from utils.loader import VLSP2016
from models.phobert import PhoBertEncoder

device = 'cuda'


if __name__ == '__main__':
    enc = PhoBertEncoder().to(device)
    dataset = VLSP2016()
    vectors = None
    labels = None

    for idx, (item, label) in enumerate(tqdm(dataset, desc='Calculating')):
        item = item.unsqueeze(0).to(device)
        label = np.array(label)

        ans = enc(item, attention_mask=(item > 0).to(device), output_hidden_states=True)
        vec = torch.cat((ans[2][-1][:, 0, ...],
                         ans[2][-2][:, 0, ...],
                         ans[2][-3][:, 0, ...],
                         ans[2][-4][:, 0, ...]), -1)

        if device == 'cuda':
            vec = vec.cpu().detach().numpy()

        vectors = np.atleast_1d(vec) if vectors is None else np.concatenate([vectors, np.atleast_1d(vec)])
        labels = np.atleast_1d(label) if labels is None else np.concatenate([labels, np.atleast_1d(label)])

    print('Vectors shape:', vectors.shape)
    print('Labels shape:', labels.shape)

    np.save(os.path.join('data', 'VLSP2016', 'vecs.npy'), vectors)
    np.save(os.path.join('data', 'VLSP2016', 'labs.npy'), labels)