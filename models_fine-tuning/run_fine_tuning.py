"""
CUDA_VISIBLE_DEVICES="1" python run_fine_tuning.py
"""

import subprocess
import os

# used to allow the import of utils_general from upper directory
import sys
sys.path.append("..")

import utils_general

MODEL_DICT = {"facebook/bart-large-cnn": "bart",
              "google/pegasus-large": "pegasus",
              "t5-large": "t5"}

def get_subprocess_cmd(output_dir, train, valid, seed, with_disfluency_tokens, model_name):
    
    cmd = f"""python ./run_summarization_no_trainer.py
               --model_name_or_path {model_name} 
               --train_file ../model_files/{train}.json 
               --validation_file ../model_files/{valid}.json 
               --output_dir {output_dir}
               --per_device_train_batch_size=4
               --per_device_eval_batch_size=4
               --seed {seed}
               --text_column text
               --summary_column summary
               --num_beams 4
               --num_train_epochs 4
               --val_min_target_length 56
               --val_max_target_length 144
               """
    
    if with_disfluency_tokens:
        cmd += "\n--with_disfluency_tokens"
        
    if "t5-large" == model_name:
        cmd += "\n--source_prefix summarize:" 
        
    cmd = cmd.split()
    
    print(cmd)
    return cmd

current_dir = os.path.join(utils_general.PATH_TO_PROJECT, "models_fine-tuning")

for model_name, model_short_name in MODEL_DICT.items():
    model_dir = os.path.join(current_dir, model_short_name)
    utils_general.create_and_or_clear_this_dir(model_dir)
    
    for seed_number in [0, 10, 20, 30, 40]:
        seed_dir = os.path.join(model_dir, "seed_"+str(seed_number))
        utils_general.create_and_or_clear_this_dir(seed_dir)

        for version in ["0", "-1", "tagged"]:
            train = f"train_{version}"
            valid = f"valid_{version}"

            output_dir = os.path.join(seed_dir, "OUTPUT__"+train+"__AND__"+valid)
            utils_general.create_and_or_clear_this_dir(output_dir)

            if "tagged" in train or "tagged" in valid:
                subprocess.run(get_subprocess_cmd(output_dir, train, valid, seed_number, True, model_name))
            else:
                subprocess.run(get_subprocess_cmd(output_dir, train, valid, seed_number, False, model_name))
