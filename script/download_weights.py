#!/usr/bin/env python
from huggingface_hub import HfFolder
import os
import shutil
from peft import PeftModel, PeftConfig
import torch
#from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PEFT_MODEL_ID = "rjac/flan-t5-xxl-senza-LoRA-qa"
CACHE_DIR = 'weights'

if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)
os.makedirs(CACHE_DIR)

HfFolder().save_token(token="")
config = PeftConfig.from_pretrained(PEFT_MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,torch_dtype=torch.float16, load_in_8bit=True,  device_map={'':0},cache_dir=CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,device_map={'':0},cache_dir=CACHE_DIR)
model = PeftModel.from_pretrained(model, PEFT_MODEL_ID,device_map={'':0},cache_dir=CACHE_DIR)