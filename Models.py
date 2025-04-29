import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self, num_classes=2):
        super(BasicModel, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.linear1 = nn.Linear(768, 384)
        self.linear2 = nn.Linear(384, num_classes)

    def forward(self, input_ids):
        features = self.linear1(input_ids)
        features = self.dropout(features)
        logits = self.linear2(features) 
        return logits