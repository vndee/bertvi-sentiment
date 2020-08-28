import os
from transformers import RobertaConfig, BertModel
from transformers import RobertaForSequenceClassification


class PhoBertForSequenceClassification():
    def __init__(self, num_classes):
        self.config = RobertaConfig.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'config.json')
        )

        self.config.num_labels = num_classes
        self.net = RobertaForSequenceClassification.from_pretrained(
            os.path.join(os.getcwd(), 'pretrained', 'PhoBERT_base_transformers', 'model.bin'),
            config=self.config,
        )

    def __call__(self, x, attention_mask=None, labels=None):
        preds = self.net(x, attention_mask=attention_mask)
        return preds

