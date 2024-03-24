import subprocess
import os

# allows the import of utils files from the upper directory
import sys
sys.path.append("..")

import utils_general

def get_subprocess_cmd(input_path, output_path):
    
    cmd = f"""python main.py 
              --input-path {input_path} 
              --output-path {output_path}
              --model ./model/swbd_fisher_bert_Edev.0.9078.pt"""
        
    cmd = cmd.split()
    
    print(cmd)
    return cmd

utils_general.create_and_or_clear_this_dir(os.path.join(utils_general.PATH_TO_PROJECT, "spotify_train", "annotated"))

f = 0.01
for n in range(0,10+1):
    
    input_path = os.path.join(utils_general.PATH_TO_PROJECT, "spotify_train", "percent_"+str(f)+"_random_state_"+str(n)) 
    output_path = os.path.join(utils_general.PATH_TO_PROJECT, "spotify_train", "annotated", "percent_"+str(f)+"_random_state_"+str(n))
    
    utils_general.just_create_this_dir(output_path)
    
    subprocess.run(get_subprocess_cmd(input_path, output_path))
    