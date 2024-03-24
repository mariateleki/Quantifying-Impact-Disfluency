"""
implementation:         https://huggingface.co/docs/transformers/model_doc/bart
bart-base model card:   https://huggingface.co/facebook/bart-large-cnn
paper:                  https://arxiv.org/abs/1910.13461
"""

import os
import shutil
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

# allows import of utils files from upper directory
import sys
sys.path.append("..")

import utils_general

# print out conda env currently in use for logging (may not actually be the default)
print("conda environment =", os.environ['CONDA_DEFAULT_ENV'])

# global variables
GPU = 0 

def initialize(fine_tuned_model_path):
    
    device = "cuda:"+str(GPU) if torch.cuda.is_available() else "cpu"
    
    if model_name == "bart":
        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
        model = BartForConditionalGeneration.from_pretrained(fine_tuned_model_path).to(device)
        
    elif model_name == "t5":
        tokenizer = T5Tokenizer.from_pretrained(fine_tuned_model_path, model_max_length=1024)
        model = T5ForConditionalGeneration.from_pretrained(fine_tuned_model_path).to(device)
    
    elif model_name == "pegasus":
        tokenizer = PegasusTokenizer.from_pretrained(fine_tuned_model_path)
        model = PegasusForConditionalGeneration.from_pretrained(fine_tuned_model_path).to(device)
    
    else:
        assert False
        
    return tokenizer, device, model

def get_summary(model_name, transcript_text):
    
    if model_name == "bart":
        inputs = tokenizer([transcript_text], max_length=1024, truncation=True, padding="longest", return_tensors="pt").to(device)
        summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=56, max_length=144)
        summary_text = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    elif model_name == "t5":
        task_prefix = "summarize: "
        input_ids = tokenizer(task_prefix+transcript_text, max_length=1024, truncation=True, padding="longest", return_tensors="pt").input_ids.to(device)
        outputs = model.generate(input_ids, min_length=56, max_length=144)
        summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    elif model_name == "pegasus":
        batch = tokenizer([transcript_text], max_length=1024, truncation=True, padding="longest", return_tensors="pt").to(device)
        translated = model.generate(**batch, min_length=56, max_length=144)
        summary_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    
    else:
        assert False
        
    return summary_text
    

for model_name in  ["bart", "t5", "pegasus"]: 
    for seed_number in [0, 10, 20, 30, 40]: 
        for train_version in ["0", "-1", "tagged"]:
            for test_version in ["0", "-1", "tagged"]:
                
                print("model_name:", model_name, "\nseed_number:", seed_number, "\ntrain_version:", train_version, "\ntest_version:", test_version)

                train = f"train_{train_version}"
                valid = f"valid_{train_version}"
                test = f"test_{test_version}"

                TEST_PATH = os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test", test_version)
                OUTPUT_PATH = os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test", test_version, f"fine-tuned__{model_name}", f"seed_{seed_number}__{train}__{test}")
                MODEL_PATH = os.path.join(utils_general.PATH_TO_PROJECT, "models_fine-tuning", model_name, f"seed_{seed_number}", "OUTPUT__"+train+"__AND__"+valid)

                # create the OUTPUT_PATH directories
                utils_general.just_create_this_dir(os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test", test_version, f"fine-tuned__{model_name}"))
                utils_general.create_and_or_clear_this_dir(os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test", test_version, f"fine-tuned__{model_name}", f"seed_{seed_number}__{train}__{test}"))

                # initialize
                tokenizer, device, model = initialize(MODEL_PATH)

                # make sure to filter out any IDE files from the directory
                files = [f for f in os.listdir(TEST_PATH) if (f.endswith(".txt") and not f.endswith("-checkpoint.txt"))]
                for current_file in tqdm(files):

                    # read the transcript text
                    transcript_text = ""
                    transcript_text = utils_general.read_file(os.path.join(TEST_PATH, current_file))

                    # do the summary
                    summary_text = get_summary(model_name, transcript_text)

                    # write the summaries out TO current_model_dir
                    utils_general.write_file(current_file.replace(".txt", "-summary.txt"), OUTPUT_PATH, summary_text)

                print()

