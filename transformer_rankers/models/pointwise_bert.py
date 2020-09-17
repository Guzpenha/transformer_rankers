from transformers import modeling_bert
from transformer_rankers.models.losses import label_smoothing
from torch import nn
from IPython import embed

class BertForPointwiseLearning(modeling_bert.BertPreTrainedModel):
    """
    BERT based model for pointwise learning to rank. It is almost identical to
    huggingface's BertForSequenceClassification, for the case when num_labels >1 
    (classification).
    """
    def __init__(self, config, loss_function="cross-entropy", smoothing=0.1):
        super().__init__(config)

        #There should be at least relevant and non relevant options (>2).
        self.num_labels = config.num_labels

        self.bert = modeling_bert.BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss()
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct = label_smoothing.LabelSmoothingCrossEntropy(smoothing)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        #forward pass for positive instances
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:            
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output