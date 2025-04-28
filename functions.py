import random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, set_seed as set_seed_transformers, BertModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    set_seed_transformers(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def get_model_tokenizer(model_id = "lanwuwei/GigaBERT-v3-Arabic-and-English"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    return model, tokenizer