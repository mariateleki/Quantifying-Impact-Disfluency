"""
implementation:         https://huggingface.co/docs/transformers/main/model_doc/pegasus
model card:             https://huggingface.co/google/pegasus-large
paper:                  https://arxiv.org/pdf/1912.08777.pdf
"""

import os
import shutil
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

# allows import of utils files from upper directory
import sys
sys.path.append("..")

import utils_general

# global variables
GPU = 1
MODEL_NAME = "pegasus"

# print out conda env currently in use for logging (may not actually be the default)
print("conda environment =", os.environ['CONDA_DEFAULT_ENV'])

# initialize
model_name = "google/pegasus-large"
device = "cuda:"+str(GPU) if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

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
        batch = tokenizer([transcript_text], max_length=1024, truncation=True, padding="longest", return_tensors="pt").to(device)
        translated = model.generate(**batch, min_length=56, max_length=144)
        summary_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

        # write the summaries out TO current_model_dir
        utils_general.write_file(current_file.replace(".txt", "-summary.txt"), current_model_dir, summary_text)

    print()



            
                

                
