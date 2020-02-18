import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import BertLMPredictionHead, BertOnlyMLMHead


class BertForMlmWithClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.lm_layer = BertLMPredictionHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        class_labels,
        masked_lm_labels=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        loss_fct = CrossEntropyLoss()

        mlm_loss, mlm_scores = 0, None
        if masked_lm_labels is not None:
            mlm_scores = self.lm_layer(sequence_output)
            mlm_loss += loss_fct(
                mlm_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)
            )
        cls_scores = self.classifier(self.dropout(pooled_output))
        cls_loss = loss_fct(cls_scores.view(-1, self.config.num_labels), class_labels.view(-1))
        return mlm_loss + cls_loss, mlm_scores, cls_scores

