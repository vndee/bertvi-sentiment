import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_optimizer(net, opts, len_data):
    param_optimizer = list(net.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=float(opts.learning_rate),
                      betas=(0.9, 0.99),
                      correct_bias=True)

    number_of_optimization_steps = int(opts.epochs * len_data / opts.batch_size / opts.accumulation_steps)
    linear_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                       num_warmup_steps=100,
                                                       num_training_steps=number_of_optimization_steps)
    constant_scheduler = get_constant_schedule(optimizer)

    return optimizer, linear_scheduler, constant_scheduler
