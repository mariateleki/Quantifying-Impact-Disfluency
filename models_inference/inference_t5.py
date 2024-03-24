"""
implementation:    https://huggingface.co/docs/transformers/model_doc/t5     
model card:        https://huggingface.co/t5-large
paper:             http://jmlr.org/papers/v21/20-074.html
"""

import os
import shutil
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

# allows import of utils files from upper directory
import sys
sys.path.append("..")

import utils_general

# global variables
GPU = 0
MODEL_NAME = "t5"

# print out conda env currently in use for logging (may not actually be the default)
print("conda environment =", os.environ['CONDA_DEFAULT_ENV'])

# initialize
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=1024)
device = "cuda:"+str(GPU) if torch.cuda.is_available() else "cpu"
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# iterate & summarize
for t in utils_general.TRF_LIST:
    
    print("current trf:", t)
    current_trf_dir = os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test", t)
    
    current_model_dir = os.path.join(current_trf_dir, MODEL_NAME)
    utils_general.create_and_or_clear_this_dir(current_model_dir)

    # make sure to filter out any IDE files from the directory
    files = [f for f in os.listdir(current_trf_dir) if (f.endswith(".txt") and not f.endswith("-checkpoint.txt"))]
    for current_file in tqdm(files):

        # read the transcript text from the current file, read FROM current_trf_dir
        transcript_text = ""
        transcript_text = utils_general.read_file(os.path.join(current_trf_dir, current_file))
            
        # do the summary
        task_prefix = "summarize: "
        input_ids = tokenizer(task_prefix+transcript_text, max_length=1024, truncation=True, padding="longest", return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids, min_length=56, max_length=144)
        summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # write the summaries out TO current_model_dir
        utils_general.write_file(current_file.replace(".txt", "-summary.txt"), current_model_dir, summary_text)

    print()

