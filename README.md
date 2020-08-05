<p align="center"> 
<h2>BERTvi-sentiment</h2>

Official repository for paper "Fine-tuning BERT-based Pre-Trained Language Models for Vietnamese Sentiment Analysis".
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/vndee/bertvi-sentiment/master/imgs/_nics2020.svg">
  <p align="center">Fine-tuning pipeline for Vietnamese sentiment analysis.</p>
</p>

This project shows how BERT-based pre-trained language models improves performance of sentiment analysis in several
Vietnamese benchmarks.  

### Requirements

- PyTorch
- Transformers
- Fairseq
- VnCoreNLP
- FastBPE

To install all dependencies:

    pip install -r requirements.txt

Download VnCoreNLP and word segmenter:

    mkdir -p vncorenlp/models/wordsegmenter
    wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
    wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
    wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
    mv VnCoreNLP-1.1.1.jar vncorenlp/ 
    mv vi-vocab vncorenlp/models/wordsegmenter/
    mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
    
Download PhoBERT pretrained model:
- PhoBERT-base:

        wget https://public.vinai.io/PhoBERT_base_transformers.tar.gz
        tar -xzvf PhoBERT_base_transformers.tar.gz

- PhoBERT-large:

        wget https://public.vinai.io/PhoBERT_large_transformers.tar.gz
        tar -xzvf PhoBERT_large_transformers.tar.gz
    
### Training

Define your own config variables in config file. 

| Variable  | Description  | Default  |
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

Training your model:

    python train.py -f config/phobert_vlsp_2016.yaml
    
All outputs will be placed at `outputs` directory.

### References

- [1] Nguyen, Dat & Nguyen, Anh. (2020). PhoBERT: Pre-trained language models for Vietnamese. Arxiv preprint: https://arxiv.org/abs/2003.00744
- [2] PhoBERT Sentiment Classification. https://github.com/suicao/PhoBert-Sentiment-Classification