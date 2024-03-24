import os
import pandas as pd

import utils_general
import utils_podcasts

def write_json(input_dir, output_dir, json_filename, testset_or_trainset):
    
    # set up the df that will be built
    d = []
    
    files_list = [file for file in os.listdir(input_dir) if file.endswith(".txt")]
    for f in files_list:
        
        # the transcript is the TEXT, the description is the SUMMARY
        transcript = utils_general.read_file(os.path.join(input_dir, f))
        description = utils_podcasts.get_description_for_episode_id(episode_id=f.replace(".txt",""), testset_or_trainset=testset_or_trainset)
        
        d.append({"text":transcript, "summary":description})
       
    # convert to pd.DataFrame
    df = pd.DataFrame(d)
    
    # write it out to file
    out = os.path.join(output_dir, json_filename+".json")
    utils_general.just_create_this_dir(output_dir)
    utils_general.delete_file_if_already_exists(out)
    df.to_json(out, orient="records", lines=True) # .dropna().to_json
        
