import torch.nn as nn

from transformers import AutoModel


MODEL_NAME = "bert-base-uncased"


class BertClassifier(nn.Module):

    def __init__(
        self,
        num_classes,
        model_name,
        dropout_rate=0.3
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(dropout_rate)

        self.classifier = nn.Linear(
            self.bert.config.hidden_size,
            num_classes
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_token = outputs.last_hidden_state[:, 0, :]

        x = self.dropout(cls_token)

        logits = self.classifier(x)

        return logits