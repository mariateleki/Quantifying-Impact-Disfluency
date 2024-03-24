import json
import os
import pandas as pd
import numpy as np
import string

from urlextract import URLExtract

import utils_general
import utils_transformations
import utils_podcasts

import utils_general

TESTSET_METADATA_DF = pd.read_csv(utils_general.PATH_TO_2020_TESTSET_DF, sep="\t")
TRAINSET_METADATA_DF = pd.read_csv(utils_general.PATH_TO_TRAINSET_DF, sep="\t")

def clean_urls(text):
    extractor = URLExtract()
    urls = extractor.find_urls(text)
    for url in urls:
        text = text.replace(url, "")
    return text

def get_description_for_episode_id(episode_id, testset_or_trainset):
    # get the description from the appropriate dataframe
    if testset_or_trainset == "testset":
        description_series = TESTSET_METADATA_DF.loc[TESTSET_METADATA_DF["episode_filename_prefix"] == episode_id, "episode_description"]
        description = str(description_series.values[0])
    else:  # testset_or_trainset == "trainset"
        description_series = TRAINSET_METADATA_DF.loc[TRAINSET_METADATA_DF["episode_filename_prefix"] == episode_id, "episode_description"]
        description = str(description_series.values[0])
        
    # clean the descriptions the same way as the transcripts
    description = clean_urls(description)
    description = description.encode("ascii", "ignore").decode()
    
    return description

# modified from https://github.com/potsawee/podcast_trec2020/blob/main/data/processor.py
# also performs some basic cleaning
def get_shortened_transcript_text_from_json_asr_file(json_asr_file, seconds):
    shortened_transcript_list = []
    with open(json_asr_file) as f:
        transcript_dict = json.loads(f.read())
        
        results_list = [r for r in transcript_dict["results"]]
        last_result = results_list[-1]
        
        for word_dict in last_result["alternatives"][0]["words"]:
                # print(word_dict)
                end_time = float(word_dict["endTime"].replace("s",""))
                word = word_dict["word"]
                if end_time <= float(seconds):
                    shortened_transcript_list.append(word)
                    
        transcript_string = " ".join(shortened_transcript_list)
        
        # clean the transcripts the same way as the descriptions
        transcript_string = clean_urls(transcript_string)
        transcript_string = transcript_string.encode("ascii", "ignore").decode()
        
        return transcript_string
    
def write_transformed_test_set_files_out():
    print("writing flattened files out...")
    
    rng = np.random.default_rng(seed=0)
    
    current_input_dir = os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test", "1min")
    
    # n = 0
    zero_dir = os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test", "0")
    utils_general.just_create_this_dir(zero_dir)

    # iterate through all the files for this transformation type
    files_list = [file for file in os.listdir(current_input_dir) if file.endswith(".txt")]
    for file in files_list:

        # open the file and get the transcript text
        transcript_text = ""
        full_filepath = os.path.join(current_input_dir, file)
        with open(full_filepath) as f:
            transcript_text = f.read()

        transcript_text = utils_transformations.get_repeats_text(0, transcript_text, rng)
        utils_general.write_file(file, zero_dir, transcript_text) 

        for t in ["repeats","interjections","false-starts","repeats-and-false-starts","repeats-and-interjections","interjections-and-false-starts","all-3"]:
            current_trf_dir = os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test", t)
            utils_general.just_create_this_dir(current_trf_dir)

            # n = 1 to 10
            for n in [1,2,3,4,5,6,7,8,9,10]:

                # open the file and get the transcript text
                transcript_text = ""
                full_filepath = os.path.join(current_input_dir, file)
                with open(full_filepath) as f:
                    transcript_text = f.read()

                # get the new filename
                new_filename = str(n) + "_" + file

                if t == "repeats":
                    transcript_text = utils_transformations.get_repeats_text(n, transcript_text, rng)

                elif t == "interjections":
                    transcript_text = utils_transformations.get_interjections_text(n, transcript_text, rng)  

                elif t == "false-starts":
                    transcript_text = utils_transformations.get_false_starts_text(n, transcript_text, rng)

                elif t == "repeats-and-false-starts":
                    transcript_text = utils_transformations.get_repeats_text(n, transcript_text, rng)
                    transcript_text = utils_transformations.get_false_starts_text(n, transcript_text, rng)

                elif t == "repeats-and-interjections":
                    transcript_text = utils_transformations.get_repeats_text(n, transcript_text, rng)
                    transcript_text = utils_transformations.get_interjections_text(n, transcript_text, rng) 

                elif t == "interjections-and-false-starts":
                    transcript_text = utils_transformations.get_interjections_text(n, transcript_text, rng)
                    transcript_text = utils_transformations.get_false_starts_text(n, transcript_text, rng)

                elif t == "all-3":
                    transcript_text = utils_transformations.get_interjections_text(n, transcript_text, rng)
                    transcript_text = utils_transformations.get_false_starts_text(n, transcript_text, rng)
                    transcript_text = utils_transformations.get_repeats_text(n, transcript_text, rng)  

                utils_general.write_file(new_filename, current_trf_dir, transcript_text)                

def write_original_test_set_files_out(directory=utils_general.PATH_TO_2020_TESTSET_DIR):
    # iterate throught the original test set podcasts
    
    out_dir = os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test", "1min")
    utils_general.create_and_or_clear_this_dir(os.path.join(utils_general.PATH_TO_PROJECT, "spotify_test"))
    utils_general.create_and_or_clear_this_dir(out_dir)
        
    files_list = []
    print("writing original files out...")
    for root, dirs, files in os.walk(directory):
        for file in files:
            
            # get the full filepath for the current file
            full_filepath = os.path.join(root, file)
            files_list.append(full_filepath)

            # write 1min files out
            transcript_text = get_shortened_transcript_text_from_json_asr_file(full_filepath, 60.0)
            
            # add a period to the end of the transcript if there isn't punctuation at the end of it already
            if transcript_text: 
                if transcript_text[-1] not in string.punctuation:
                    transcript_text += "."
            
            new_filepath = os.path.join(out_dir, file.replace(".json", ".txt"))
            utils_general.write_file_if_not_blank(new_filepath, transcript_text)