import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

class GermanClassifier(nn.Module):
    def __init__(self, model_name, num_labels, class_weights=None, dropout_rate=0.1):
        super().__init__()
        
   
        config = AutoConfig.from_pretrained(
            model_name, 
            num_labels=num_labels,
            classifier_dropout=dropout_rate,
        )
        

        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            config=config
        )
  
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):

        global_attention_mask = None
        
        if "longformer" in self.backbone.config.model_type:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
        

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None 
        )
        
        logits = outputs.logits
        
        loss = None
        if labels is not None:
            if self.criterion.weight is not None:
                self.criterion.weight = self.criterion.weight.to(logits.device)
            loss = self.criterion(logits, labels)
            
        return {"loss": loss, "logits": logits}

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)