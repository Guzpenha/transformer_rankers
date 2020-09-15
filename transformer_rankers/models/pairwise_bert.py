from transformers import modeling_bert
from transformer_rankers.models.losses import label_smoothing
from torch import nn
from IPython import embed

class BertForPairwiseLearning(modeling_bert.BertPreTrainedModel):
    """
    BERT based model for pairwise learning. It expects both the <q, positive_doc> and the <q, negative_doc>
    for doing the forward pass. The loss is cross-entropy for the difference between positive_doc and negative_doc
    scores (labels are 1 if score positive_neg > score negative_doc otherwise 0) based on 
    "Learning to Rank using Gradient Descent" 2005 ICML.
    """
    def __init__(self, config, loss_function="cross-entropy"):
        super().__init__(config)

        #There should be at least relevant and non relevant options.
        self.num_labels = config.num_labels
        self.bert = modeling_bert.BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss()
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct = label_smoothing.LabelSmoothingCrossEntropy()

        self.init_weights()

    def forward(
        self,
        input_ids_pos=None,
        attention_mask_pos=None,
        token_type_ids_pos=None,
        input_ids_neg=None,
        attention_mask_neg=None,
        token_type_ids_neg=None,
        labels=None
    ):
        #forward pass for positive instances
        outputs_pos = self.bert(
            input_ids_pos,
            attention_mask=attention_mask_pos,
            token_type_ids=token_type_ids_pos
        )
        pooled_output_pos = outputs_pos[1]
        pooled_output_pos = self.dropout(pooled_output_pos)
        logits_pos = self.classifier(pooled_output_pos)

        #forward pass for negative instances
        outputs_neg = self.bert(
            input_ids_neg,
            attention_mask=attention_mask_neg,
            token_type_ids=token_type_ids_neg
        )
        pooled_output_neg = outputs_neg[1]
        pooled_output_neg = self.dropout(pooled_output_neg)
        logits_neg = self.classifier(pooled_output_neg)

        logits_diff = logits_pos - logits_neg

        #Calculating Cross entropy loss for pairs <q,d1,d2>
        # based on "Learning to Rank using Gradient Descent" 2005 ICML
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits_diff.view(-1, self.num_labels), labels.view(-1))

        output = (logits_pos,) + outputs_pos[2:]
        return ((loss,) + output) if loss is not None else output