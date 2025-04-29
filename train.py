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
from sklearn.metrics import classification_report, accuracy_score
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
from Models import *
from sklearn.svm import SVC
from batching import *
import wandb

logging.set_verbosity_error()

set_seed(42)
    
def run_all(args):
    le= LabelEncoder()
    device = args.device
    data_df = download_prepare_dataset()
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

    
    bertmodel, tokenizer = get_model_tokenizer(args.model_id)
    

    for param in bertmodel.parameters():
        param.requires_grad = False

    if args.model_type.lower()=='basic':
        run = wandb.init(entity="omaratef3221", project="Fake News Project - RBF", 
                         name = "Fake News Normal Head",
                         config={
                             "learning_rate": args.lr, "architecture": 
                             "BERT with Dense Classification Head", 
                             "dataset": "SANAD: Single-Label Arabic News Articles Dataset",
                             "epochs": args.EPOCHS, 
                             "batch_size": args.BATCH_SIZE})
        model = BasicModel(len(data_df['category'].unique())).to(device)
        bertmodel.to(device)
        print_number_of_trainable_model_parameters(model)
        train_normal(args, data_df, device, le, model, tokenizer, run, bertmodel)

    if args.model_type.lower()=='rbf':
        kernel = 'rbf'
        degree= 5
        gamma='auto'
        run = wandb.init(entity="omaratef3221", project="Fake News Project - RBF", 
                         name = "Fake News SVM Head",
                         config={
                             "kernel": kernel, 
                             "architecture": "BERT with SVM Classification Head", 
                             "dataset": "SANAD: Single-Label Arabic News Articles Dataset",
                             "degree": degree,
                             "gamma": gamma
                             })

        X_train, X_test, y_train, y_test = train_test_split(np.array(data_df["text"]), np.array(data_df["category"]), test_size=0.3, stratify = data_df['category'].values)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify = y_test)
        print("Training set is: ", X_train.shape[0], " rows which is ", round(X_train.shape[0]/data_df.shape[0],4)*100, "%")
        print("Validation set is: ",X_val.shape[0], " rows which is ", round(X_val.shape[0]/data_df.shape[0],4)*100, "%")
        print("Testing set is: ",X_test.shape[0], " rows which is ", round(X_test.shape[0]/data_df.shape[0],4)*100, "%")
        bertmodel.to(device)
        X_train = process_data_in_batches(bertmodel, tokenizer, X_train, args.BATCH_SIZE, device)
        X_val = process_data_in_batches(bertmodel, tokenizer, X_val, args.BATCH_SIZE, device)
        X_test = process_data_in_batches(bertmodel, tokenizer, X_test, args.BATCH_SIZE, device)
        X_train = X_train.cpu().numpy()
        X_val = X_val.cpu().numpy()
        X_test = X_test.cpu().numpy()
        clf = SVC(kernel = kernel, degree= degree, gamma=gamma)
        t1 = time.time()
        clf.fit(X_train, y_train)
        train_time =  time.time() - t1
        print("Time Taken to train RBF: ",train_time)
        t1 = time.time()
        predictions = clf.predict(X_test)
        test_time =  time.time() - t1
        class_report= classification_report(y_test, predictions, target_names=le.classes_, output_dict=True)
        
        print("Accuracy Score: ", accuracy_score(y_test, predictions))
        report_columns =  ["Class", "Precision", "Recall", "F1-score", "Support"]
        for class_name, metrics in class_report.items():
            if class_name in le.classes_:  # Filter to actual class names only
                wandb.log({
                    f"{class_name}/Precision": metrics["precision"],
                    f"{class_name}/Recall": metrics["recall"],
                    f"{class_name}/F1-score": metrics["f1-score"],
                    f"{class_name}/Support": metrics["support"]
                })
        run.log({"Overall Accuracy Score": accuracy_score(y_test, predictions)})
        run.log({"Training time: ": train_time})
        run.log({"Testing/Inference time: ": test_time})
        class_report= classification_report(y_test, predictions, target_names=le.classes_)
        print(class_report)
        class_report = class_report.splitlines()
        
        report_table = []
        for line in class_report[2:(len(le.classes_)+2)]:
            report_table.append(line.split())
        run.log({
            "SVM Classification Report": wandb.Table(data=report_table, columns=report_columns),})
        

def train_normal(args, data_df, device, le, model, tokenizer, run, bertmodel):
    testing_predictions = []
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr= args.lr)

    X_train, X_test, y_train, y_test = train_test_split(np.array(data_df["text"]), np.array(data_df["category"]), test_size=0.3, stratify = data_df['category'].values)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify = y_test)
    print("Training set is: ", X_train.shape[0], " rows which is ", round(X_train.shape[0]/data_df.shape[0],4)*100, "%")
    print("Validation set is: ",X_val.shape[0], " rows which is ", round(X_val.shape[0]/data_df.shape[0],4)*100, "%")
    print("Testing set is: ",X_test.shape[0], " rows which is ", round(X_test.shape[0]/data_df.shape[0],4)*100, "%")

    X_train = process_data_in_batches(bertmodel, tokenizer, X_train, args.BATCH_SIZE, device)
    X_val = process_data_in_batches(bertmodel, tokenizer, X_val, args.BATCH_SIZE, device)
    X_test = process_data_in_batches(bertmodel, tokenizer, X_test, args.BATCH_SIZE, device)

    training_data = dataset(X_train, y_train, tokenizer, device)
    validation_data = dataset(X_val, y_val, tokenizer, device)
    testing_data = dataset(X_test, y_test, tokenizer, device)

    train_dataloader = DataLoader(training_data, batch_size=args.BATCH_SIZE, shuffle= False)
    validation_dataloader = DataLoader(validation_data, batch_size=args.BATCH_SIZE, shuffle= False)
    testing_dataloader = DataLoader(testing_data, batch_size=args.BATCH_SIZE, shuffle= False)

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
            labels = labels.to(device)
            logits = model(inputs) 
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
                labels = labels.to(device)
                logits = model(inputs)
                loss_val = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                acc = (preds == labels).sum().item()

                total_loss_val += loss_val.item()
                total_acc_val += acc

        total_loss_train = total_loss_train / len(train_dataloader)
        total_loss_val = total_loss_val / len(validation_dataloader)

        total_acc_train = total_acc_train / len(training_data) * 100
        total_acc_val = total_acc_val / len(validation_data) * 100

        total_loss_train_plot.append(round(total_loss_train, 4))
        total_loss_validation_plot.append(round(total_loss_val, 4))
        total_acc_train_plot.append(round(total_acc_train, 4))
        total_acc_validation_plot.append(round(total_acc_val, 4))

        run.log({"epoch": epoch, "train-loss": total_loss_train, "train accuracy": total_acc_train, "val loss": total_loss_val, "val accuracy": total_acc_val})
        print(f"Epoch {epoch+1}: Train Loss = {total_loss_train:.4f}, Train Acc = {total_acc_train:.2f}%, Val Loss = {total_loss_val:.4f}, Val Acc = {total_acc_val:.2f}%")
        print("=" * 50)
    train_time = time.time()-t1
    print(f"Training Time: {train_time:.2f} seconds")
    print("Training Complete")
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
            prediction = model(input)
            batch_loss_val = criterion(prediction, label)
            total_loss_test += batch_loss_val.item()
            acc = (prediction.argmax(dim=1).round() == label).sum().item()
            testing_predictions += prediction.argmax(dim=1).round().cpu().numpy().tolist()
            total_acc_test += acc
    
    test_time = time.time()-t1
    print(f"Testing Time: {test_time:.2f} seconds")
    print(f"Accuracy Score is: {round((total_acc_test/X_test.shape[0])*100, 2)}%")
    print("Classification Report: ")
    class_report= classification_report(y_test, testing_predictions, target_names=le.classes_, output_dict=True)
    
    print("Accuracy Score: ", accuracy_score(y_test, testing_predictions))
    for class_name, metrics in class_report.items():
        if class_name in le.classes_:  # Filter to actual class names only
            wandb.log({
                f"{class_name}/Precision": metrics["precision"],
                f"{class_name}/Recall": metrics["recall"],
                f"{class_name}/F1-score": metrics["f1-score"],
                f"{class_name}/Support": metrics["support"]
            })
    run.log({"Overall Accuracy Score": accuracy_score(y_test, testing_predictions)})
    run.log({"Training time: ": train_time})
    run.log({"Testing/Inference time: ": test_time})

    report_columns =  ["Class", "Precision", "Recall", "F1-score", "Support"]
    class_report= classification_report(y_test, testing_predictions, target_names=le.classes_)
    print(class_report)
    class_report = class_report.splitlines()
    report_table = []
    for line in class_report[2:(len(le.classes_)+2)]:
        report_table.append(line.split())
    run.log({
        "Normal Head Classification Report": wandb.Table(data=report_table, columns=report_columns)
        })
    
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
    run_all(args)