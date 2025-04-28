import kagglehub
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

def download_prepare_dataset():
    path = kagglehub.dataset_download("haithemhermessi/sanad-dataset")
    categories = []
    texts = []
    for category in os.listdir(path):
        for text in os.listdir(os.path.join(path, category)):
            categories.append(category)
            f = open(os.path.join(path, category, text), "r")
            texts.append(f.read())
    data = pd.DataFrame({"text": texts, "category": categories})
    return data

class dataset(Dataset):
    def __init__(self, X, Y, tokenizer, device):
        self.X = [tokenizer(x,
                            max_length = 150,
                            truncation = True,
                            padding = 'max_length',
                            return_tensors='pt').to(device)
                for x in X
                ]

        self.Y = torch.tensor(Y, dtype = torch.long).to(device)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    