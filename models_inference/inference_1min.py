# For inference, we simply copy over the 1min files in the same format as the summarization models (which actually do inference). We don't actually do any inference for the 1min baseline. This is simply a baseline, following the TREC 2020 baseline.

import os
import shutil
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

# allows import of utils files from upper directory
import sys
sys.path.append("..")

import utils_general

# global variables
MODEL_NAME = "1min"

# print out conda env currently in use for logging (may not actually be the default)
print("conda environment =", os.environ['CONDA_DEFAULT_ENV'])

# iterate & summarize
for t in utils_general.TRF_LIST:
    
    print("current trf:", t)
    current_trf_dir = os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test", t)
    
    current_model_dir = os.path.join(current_trf_dir, MODEL_NAME)
    utils_general.create_and_or_clear_this_dir(current_model_dir)

    # make sure to filter out any IDE files from the directory
    files = [f for f in os.listdir(current_trf_dir) if (f.endswith(".txt") and not f.endswith("-checkpoint.txt"))]
    for current_file in tqdm(files):
        
        summary_text = ""
        summary_text = utils_general.read_file(os.path.join(current_trf_dir, current_file))

        # write the summaries out TO current_model_dir
        utils_general.write_file(current_file.replace(".txt", "-summary.txt"), current_model_dir, summary_text)

    print()
