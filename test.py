## 测试


import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import re

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from utils.prompter import Prompter, ZeroPrompter, SimilarityPrompter
from sklearn.metrics import  f1_score,precision_score,recall_score,accuracy_score,roc_auc_score,accuracy_score

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel
)

def getmetric(labels,predictions):
    f1 = f1_score(labels, predictions)
    recall = recall_score(labels, predictions)
    precision = precision_score(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    acc = accuracy_score(labels,predictions)
    print("f1 = {},recall = {}, precision = {}, auc = {}, acc = {}".format(f1,recall,precision,auc,acc))

pred_lst = []
ground = []
peft_model_path = "model/baichuan-v2-7b-recall"
model_path = "../pre_trained_model/baichuan-v2-7b"
dataset = load_dataset('json', data_files={'test':'data/search_chat/query_12.json'})

accelerator_log_kwargs = {}
accelerator = Accelerator(gradient_accumulation_steps=1, **accelerator_log_kwargs)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to('cuda')

if peft_model_path != None and peft_model_path != "":
    model = PeftModel.from_pretrained(model,peft_model_path)

from transformers import DataCollatorForSeq2Seq
class testDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        inputs = [feature["input"] for feature in features] if "input" in features[0].keys() else None
        
        features = self.tokenizer(
            inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        ).to('cuda')
        return features

tokenizer.padding_side = "left"
batch_size = 8
data_collator = testDataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=4, return_tensors="pt", padding=True,max_length=64)
test_dataloader = DataLoader(dataset['test'], collate_fn=data_collator, batch_size=batch_size)

accelerator.wait_for_everyone()

test_dataloader = accelerator.prepare(test_dataloader)

if accelerator.is_local_main_process:
    print("****start model inference****")
    print("model test step = ",len(test_dataloader))

pbar = tqdm(range(len(test_dataloader)), disable=not accelerator.is_local_main_process,desc="Example")

res = []

for step, batch in enumerate(test_dataloader):
    with torch.no_grad():
        pred = model.generate(**batch, max_new_tokens=20,num_beams=5,num_return_sequences=2)
        
    pred = accelerator.pad_across_processes(pred, dim=1, pad_index=0)
    all_preds = accelerator.gather(pred)
    if accelerator.is_local_main_process:
        for i in range(all_preds.shape[0]):
            output = tokenizer.decode(all_preds[i], skip_special_tokens=True)
            if output != None:
                res.append(output)
    pbar.update(1)

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    print("***writeing result***",len(res))
    with open ("data/search_chat/model_result1/model_output_12","w") as wf:
        for pred in res:
            wf.write(pred + "\n")
