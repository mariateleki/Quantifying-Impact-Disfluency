import json
import os
import pandas as pd
import evaluate
from tqdm import tqdm
import time

import utils_general
import utils_podcasts

def get_episode_id_and_N_parameter(f, slow_mode):
    
    full_filename = f
    f = f.split("/")[-1]
    
    # only want to calculate rouge for summary files, so making sure that's what's happening
    assert "-summary" in f
    
    # then safe to filter out the "-summary"
    if "-summary" in f:
        f = f.replace("-summary", "")
    
    # and pull out the episode_id and num_repeats from the filename
    if "_" in f:
        episode_id = f.split("_")[1].replace(".txt","")
        N_parameter = f.split("_")[0]
    else:
        episode_id = f.replace(".txt","")
        N_parameter = full_filename.split("/")[-2]  # this will be the .. dir's name, this makes sense because the dirs with this are: 0, -1, tagged (tagged is kinda special, but for 0 and -1 this is especially useful)
    
    if slow_mode:
        print("episode_id:", episode_id, "N_parameter:", N_parameter)
    
    return episode_id, N_parameter

def calculate_and_write_rouge_csv(input_dir, output_name, slow_mode, testset_or_trainset):
    
    # write out the first (header) line of the csv
    output_path = os.path.join(utils_general.PATH_TO_PROJECT, "csv", output_name)
    utils_general.delete_file_if_already_exists(output_path)
    with open(output_path, mode="w+") as f:
        f.write("episode_id,N_parameter,rouge1,rouge2,rougeL,rougeLsum\n")

    # only check actual files
    files_list = [f for f in os.listdir(input_dir) if (f.endswith(".txt") and not f.endswith("-checkpoint.txt"))]
    for current_file in tqdm(files_list):

        # read summary text from file
        summary_text = utils_general.read_file(os.path.join(input_dir, current_file))
        episode_id,N_parameter = get_episode_id_and_N_parameter(os.path.join(input_dir, current_file), slow_mode)

        # get episode_id description from METADATA_DF
        description = utils_podcasts.get_description_for_episode_id(episode_id, testset_or_trainset)
        
        if slow_mode:
            print("\nDESCRIPTION:", description)
            print("SUMMARY TEXT:", summary_text, "\n")

         # calculate rouge scores
        scores = evaluate_rouge.compute(references=[description], 
                                        predictions=[summary_text])

        # write out each line
        with open(output_path, mode="a") as f:
            f.write("{},{},{},{},{},{}\n".format(episode_id,
                                                 N_parameter,
                                                 scores["rouge1"],
                                                 scores["rouge2"],
                                                 scores["rougeL"],
                                                 scores["rougeLsum"]))
        
        if slow_mode:
            time.sleep(10)  # can change the number of seconds here if you need more time to read

if __name__ == "__main__":
    
    # set up the evaluate library for ROUGE scores
    evaluate_rouge = evaluate.load('rouge')
    
    # write rouge score csvs for each of the INFERENCE trfs and models
    print("utils_general.TRF_LIST :", utils_general.TRF_LIST)
    print("utils_general.INFERENCE_MODEL_LIST :", utils_general.INFERENCE_MODEL_LIST)
    
    for t in utils_general.TRF_LIST:
        print("current t :", t)
        current_trf_dir = os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test", t)
        
        for m in utils_general.INFERENCE_MODEL_LIST:
            print("current m :", t)
            current_m_dir = os.path.join(current_trf_dir, m)
            
            current_output_name = t + "__" + m + "__ROUGE_SCORES.csv"
            
            # can toggle slow_mode=True to see the description and summary text printed 1-by-1 to the terminal
            calculate_and_write_rouge_csv(input_dir=current_m_dir, output_name=current_output_name, slow_mode=False, testset_or_trainset="testset")
            
            print()
    
    # write rouge score csvs for each of the FINE-TUNED trfs and models
    for model_name in ["fine-tuned__bart", "fine-tuned__pegasus", "fine-tuned__t5"]:
        for test_version in ["tagged", "-1", "0"]:
            
            current_outer_dir = os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test", test_version, model_name)

            for seed_number in [0, 10, 20, 30, 40]:
                for train_version in ["0", "-1", "tagged"]:

                    train = f"train_{train_version}"
                    test = f"test_{test_version}"
                    m = f"seed_{seed_number}__{train}__{test}"

                    current_inner_dir = os.path.join(current_outer_dir, m)

                    current_output_name = model_name + "__" + m + "__ROUGE_SCORES.csv"
                    
                    print(current_output_name)

                    # can toggle slow_mode=True to see the description and summary text printed 1-by-1 to the terminal
                    calculate_and_write_rouge_csv(input_dir=current_inner_dir, output_name=current_output_name, slow_mode=False, testset_or_trainset="testset")

                    print()

                    