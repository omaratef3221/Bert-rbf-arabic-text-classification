from rbf_layer import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFModel(nn.Module):
    def __init__(self, bert, num_classes=2):
        super(RBFModel, self).__init__()
        self.bert = bert
        self.project = nn.Linear(768, 256)
        self.norm = nn.LayerNorm(256)
        self.rbf1 = RBFLayer(
            in_features_dim=256,
            num_kernels=64,
            out_features_dim=32,
            radial_function=rbf_gaussian,
            norm_function=l_norm
        )

        self.rbf2 = RBFLayer(
            in_features_dim=32,
            num_kernels=16,
            out_features_dim=num_classes,
            radial_function=rbf_gaussian,
            norm_function=l_norm
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
        x = self.project(cls_token)
        x = self.norm(x)  
        x = x.unsqueeze(1) 
        x = self.rbf1(x)
        logits = self.rbf2(x)
        return logits.squeeze(1)


class BasicModel(nn.Module):
    def __init__(self, bert, num_classes=2):
        super(BasicModel, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.25)
        self.linear1 = nn.Linear(768, 384)
        self.linear2 = nn.Linear(384, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        features = self.linear1(cls_token)
        features = self.dropout(features)
        logits = self.linear2(features) 
        return logits