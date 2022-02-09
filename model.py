import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerPreTrainedModel


class LongformerSoftmaxForNer(LongformerPreTrainedModel):
    def __init__(self, config):
        super(LongformerSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = torch.log_softmax(self.classifier(sequence_output), dim=-1)
        return logits
