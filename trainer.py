import argparse
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from datasets import load_from_disk,load_dataset
import evaluate
from peft import LoraConfig,get_peft_model,TaskType
import numpy as np
import torch
import os
from transformers import Trainer,TrainingArguments
import logging

def create_dataset(data_path:str):
    if os.path.exists(data_path):
        data=load_from_disk(data_path)
    else:
        logging.info(f"Loading dataset from {data_path}")
        data=load_dataset(data_path)
    return data

def bulid_model(model_path,num_labels):
    logging.info(f"Loading modeland tokenier from {model_path}")
    model=AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=num_labels)
    tokenier=AutoTokenizer.from_pretrained(model_path)
    return model,tokenier

def data_preocessing(data,tokenier,max_length=512):
    return tokenier(data['text'],truncation=True,padding=True,max_length=max_length,return_tensors='pt')

def load_mertic(metric):
    logging.info(f"loading {metric} metric")
    way=evaluate.load(metric)
    return way

def metric_comp(eval_pred,metric):
    pred,true=eval_pred
    pred=np.argmax(pred,axis=-1)
    return metric.compute(predictions=pred, references=true)

def train(out_dir,model,data,train_batch=32,val_batch=32,epochs=5):
    args=TrainingArguments(output_dir=out_dir,per_device_train_batch_size=train_batch,per_device_eval_batch_size=val_batch,num_train_epochs=epochs,eval_delay='epochs')
    trainer=Trainer(model=model,args=args,train_dataset=data['train'],eval_dataset=data['test'],compute_metrics=metric_comp)
    return trainer

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train a  text classification model.')
    parser.add_argument('--model_path', type=str,help='load_model and tokenier')
    parser.add_argument('--data_path', type=str,help='load_dataset')
    parser.add_argument('--metric_name', type=str,help='load_metric')
    parser.add_argument('--out_dir', type=str,help='output_dir')
    parser.add_argument('--train_batch', type=int,help='train_batch')
    parser.add_argument('--val_batch', type=int,help='val_batch')
    parser.add_argument('--epochs', type=int,help='train_epochs')
    args = parser.parse_args()
    data=create_dataset(args.data_path)
    num_labels=data['train'].features['label'].num_classes
    model,tokenier=bulid_model(args.model_path,num_labels=num_labels)
    data=data.map(data_preocessing,fn_kwargs={'tokenier':tokenier},batched=True,batch_size=50000)
    data['train']=data['train'].shuffle(seed=42).select(range(500))
    data['test']=data['test'].shuffle(seed=42).select(range(100))
    metric=load_mertic(args.metric_name)
    trainer=train(args.out_dir, model, data, args.train_batch, args.val_batch, args.epochs)
    trainer.train()

    