import warnings
warnings.filterwarnings("ignore") 
import pandas as pd
from clean_data import text_prepare
import argparse
from functions import *
import torch
import torch.nn as nn
import torch.nn.functional as F
print("Cuda available: ", torch.cuda.is_available())
from torch.optim import Adam, AdamW
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import os
import time
from get_dataset import download_prepare_dataset, dataset
from create_plots import create_plots
from transformers import logging
import requests
import random
import sys
from rbf_layer import *
from Models import *

logging.set_verbosity_error()


set_seed(42)
    
def train(args):
    le= LabelEncoder()
    device = args.device
    data_df = download_prepare_dataset().sample(2500)
    print("Data Distribution: ")
    print(data_df['category'].value_counts())
    data_df['category'] = le.fit_transform(data_df['category'])
    print()
    print("Classes are: ", len(le.classes_) , " which are: " ,  le.classes_)
    args.save_model_path = "./"
    if os.path.exists(args.save_model_path):
        print(f" {args.save_model_path} Directory already exists")
    else:
        os.makedirs(args.save_model_path)
        print(f"{args.save_model_path} Directory Created")
    np.save(os.path.join(args.save_model_path, "label_encoder.npy"), le.classes_)

    testing_predictions = []
    bertmodel, tokenizer = get_model_tokenizer(args.model_id)
    # print(print_number_of_trainable_model_parameters(model))
    

    for param in bertmodel.parameters():
        param.requires_grad = False

    if args.model_type.lower()=='basic':
        model = BasicModel(bertmodel, len(data_df['category'].unique())).to(device)
    if args.model_type.lower()=='rbf':
        model = RBFModel(bertmodel, len(data_df['category'].unique())).to(device)

    print_number_of_trainable_model_parameters(model)
    print()
    print("=="*30)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr= args.lr)

    X_train, X_test, y_train, y_test = train_test_split(np.array(data_df["text"]), np.array(data_df["category"]), test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
    print("Training set is: ", X_train.shape[0], " rows which is ", round(X_train.shape[0]/data_df.shape[0],4)*100, "%")
    print("Validation set is: ",X_val.shape[0], " rows which is ", round(X_val.shape[0]/data_df.shape[0],4)*100, "%")
    print("Testing set is: ",X_test.shape[0], " rows which is ", round(X_test.shape[0]/data_df.shape[0],4)*100, "%")

    training_data = dataset(X_train, y_train, tokenizer, device)
    validation_data = dataset(X_val, y_val, tokenizer, device)
    testing_data = dataset(X_test, y_test, tokenizer, device)

    train_dataloader = DataLoader(training_data, batch_size=args.BATCH_SIZE, shuffle= False)
    validation_dataloader = DataLoader(validation_data, batch_size=args.BATCH_SIZE, shuffle= False)
    testing_dataloader = DataLoader(testing_data, batch_size=args.BATCH_SIZE, shuffle= False)

    # Initialize tracking lists
    total_loss_train_plot = []
    total_loss_validation_plot = []
    total_acc_train_plot = []
    total_acc_validation_plot = []

    t1 = time.time()

    for epoch in range(args.EPOCHS):
        model.train()
        total_loss_train = 0
        total_acc_train = 0

        # Training
        for batch in train_dataloader:
            inputs, labels = batch
            inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            logits = model(inputs['input_ids'], inputs['attention_mask'])  # (batch_size, num_classes)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            acc = (preds == labels).sum().item()

            total_loss_train += loss.item()
            total_acc_train += acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total_loss_val = 0
        total_acc_val = 0

        with torch.no_grad():
            for batch in validation_dataloader:
                inputs, labels = batch
                inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                logits = model(inputs['input_ids'], inputs['attention_mask'])
                loss_val = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                acc = (preds == labels).sum().item()

                total_loss_val += loss_val.item()
                total_acc_val += acc

        avg_train_loss = total_loss_train / len(train_dataloader)
        avg_val_loss = total_loss_val / len(validation_dataloader)

        avg_train_acc = total_acc_train / len(training_data) * 100
        avg_val_acc = total_acc_val / len(validation_data) * 100

        total_loss_train_plot.append(round(avg_train_loss, 4))
        total_loss_validation_plot.append(round(avg_val_loss, 4))
        total_acc_train_plot.append(round(avg_train_acc, 4))
        total_acc_validation_plot.append(round(avg_val_acc, 4))

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {avg_train_acc:.2f}%, Val Loss = {avg_val_loss:.4f}, Val Acc = {avg_val_acc:.2f}%")
        print("=" * 50)

    print(f"Training Time: {time.time()-t1:.2f} seconds")
    print("Training Complete")
    # create_plots(total_loss_train_plot, 
    #                 total_loss_validation_plot, 
    #                 total_acc_train_plot, 
    #                 total_acc_validation_plot, 
    #                 args.EPOCHS, 
    #                 os.path.join("_plot.png"))
    # Model Save
    torch.save(model.state_dict(), os.path.join(args.save_model_path, "model.pth"))
    print("Model Saved")
    t1 = time.time()
    with torch.no_grad():
        total_loss_test = 0
        total_acc_test = 0
        for indx, data in enumerate(testing_dataloader):
            input, label = data
            input.to(device)
            label.to(device)

            prediction = model(input['input_ids'].squeeze(1), input['attention_mask'].squeeze(1)).squeeze(1)
            batch_loss_val = criterion(prediction, label)
            total_loss_test += batch_loss_val.item()
            acc = (prediction.argmax(dim=1).round() == label).sum().item()
            testing_predictions += prediction.argmax(dim=1).round().cpu().numpy().tolist()
            total_acc_test += acc
    print(f"Testing Time: {time.time()-t1:.2f} seconds")

    print(f"Accuracy Score is: {round((total_acc_test/X_test.shape[0])*100, 2)}%")
    print("Classification Report: ")
    print(classification_report(testing_data.Y.cpu().numpy() , testing_predictions, target_names=le.classes_))
    requests.post("https://ntfy.sh/rbf_bert", data=f"{args.save_model_path} is Done".encode(encoding='utf-8'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="lanwuwei/GigaBERT-v3-Arabic-and-English")
    parser.add_argument("--device", type=str, default="mps" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--BATCH_SIZE", type=int, default=16)
    parser.add_argument("--EPOCHS", type=int, default=10)
    parser.add_argument("--save_model_path", type=str, default="basic_model_path")
    parser.add_argument("--model_type", type = str, default = 'basic')
    args = parser.parse_args()
    train(args)