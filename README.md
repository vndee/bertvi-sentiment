### BERTvi-sentiment
Official repository for paper "Fine-tuning BERT-based Pre-Trained Language Models for Vietnamese Sentiment Analysis".

#### Requirements

- PyTorch
- Transformers
- Fairseq

To install all dependencies:

    pip install -r requirements.txt
    
#### Training

Define your own config variables in config file. 

| Varialble  | Description  | Default  |
|---|---|---|
| device  | Training device: `cpu` or `cuda`.  | `cuda`  |
| dataset  | Which dataset will be used for training phrase: `vlsp2016`, `aivivn`, `uit-vsfc`.  | `vlsp2016`  |
| encoder  | BERT encoder model: `phobert`, `bert`.  | `phobert`  |
| epochs  | Number of training epochs.  | `15`  |
| batch_size  | Number of sample per batch.  | `8`  |
| feature_shape  | Encoder output feature shape.  | `768`  |
| num_classes  | Number of classes.  | `3`  |
| pivot (Optional) | For splitting aivivn dataset.  | `0.8`  |
| max_length  | Max sequence length for encoder.  | `256`  |
| tokenizer_type  | Sentence tokenizer for BERT encoder: `phobert`, `bert`.  | `phobert`  |
| num_workers  | Number of worker to produce dataset.  | `4`  |
| learning_rate  | Learning rate.  | `3e-5`  |
| momentum  | Optimizer momentum.  | `0.9`  |
| random_seed  | Random seed.  | `101`  |
| accumulation_steps  | Optimizer accumulation step.  | `5`  |
| pretrained (Optional) | Pretrained model path. | `None` | 