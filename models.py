from transformers import BertPreTrainedModel, BertModel
from torch import nn

class BertNerSoftmax(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # (batch_size, seq_len, num_labels) => (batch_size * seq_len, num_labels)
        logits = self.classifier(sequence_output).view(-1, self.num_labels)

        # (batch_size, seq_len) => (batch_size*seq_len,)
        attention_mask = attention_mask.view(-1)
        labels = labels.view(-1)

        loss_fct = nn.CrossEntropyLoss()
        # != -100: other sub_token\[CLS]\[SEP]\[PAD]
        # attention_mask = 1: valid sentense
        filter_ignored = (labels != -100) & (attention_mask == 1)
        active_logits = logits[filter_ignored, :]
        active_labels = labels[filter_ignored]

        loss = loss_fct(active_logits, active_labels)
        return loss, active_logits, active_labels
