import os
import shutil

# #############################################################
# SET THESE TWO PATHS                                         #
PATH_TO_PROJECT = "YOUR_DIRECTORY/disfluency-spotify"         #
PATH_TO_SPOTIFY = "YOUR_DIRECTORY/podcasts-no-audio-13GB"     #
# #############################################################

PATH_TO_FIGS = os.path.join(PATH_TO_PROJECT, "figs")
PATH_TO_CSV = os.path.join(PATH_TO_PROJECT, "csv")

PATH_TO_SPOTIFY_TESTSET = os.path.join(PATH_TO_SPOTIFY, "TREC/spotify-podcasts-2020")
PATH_TO_2020_TESTSET_DIR = os.path.join(PATH_TO_SPOTIFY_TESTSET, "podcasts-transcripts-summarization-testset")
PATH_TO_2020_TESTSET_DF = os.path.join(PATH_TO_SPOTIFY_TESTSET, "metadata-summarization-testset.tsv")

PATH_TO_TRAINSET_DIR = os.path.join(PATH_TO_SPOTIFY, "spotify-podcasts-2020/podcasts-transcripts")
PATH_TO_TRAINSET_DF = os.path.join(PATH_TO_SPOTIFY, "metadata.tsv")

# useful for the transformations
TRF_LIST = ["0", 
            "repeats",
            "interjections",
            "false-starts",
            "interjections-and-false-starts",
            "repeats-and-false-starts",
            "repeats-and-interjections", 
            "all-3"]

INFERENCE_MODEL_LIST = ["1min", 
                        "bart", 
                        "cued_speechUniv2", 
                        "pegasus", 
                        "t5",
                        "llama"]

def write_file(new_filename, directory, text):
    new_filepath = os.path.join(directory, new_filename)
    with open(new_filepath, mode="w") as f:
        f.write(text)
        
def write_file_if_not_blank(file, text):
    if text:
        with open(file, mode="w") as f:
            f.write(text)
            
def read_file(filepath):
    text = ""
    with open(filepath) as f:
        text = f.read()
    return text

def delete_file_if_already_exists(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

def create_and_or_clear_this_dir(d):
    # check if all the necessary dirs exist, or need to be created
    if not os.path.isdir(d):
        os.mkdir(d)
    # if this dir does exist, clear it out and re-create it
    else:
        shutil.rmtree(d)
        os.mkdir(d)
        
def just_create_this_dir(d):
    # check if all the necessary dirs exist, or need to be created
    if not os.path.isdir(d):
        os.mkdir(d)
        
def copy_all_files(input_dir, output_dir):
    
    print("input_dir:", input_dir, "\noutput_dir:", output_dir, "\n")
    print("# of files in input_dir:", len(os.listdir(input_dir)))

    for file in os.listdir(input_dir):
        path_to_file = os.path.join(input_dir, file)

        text = ""
        text = utils_general.read_file(path_to_file)

        utils_general.write_file(new_filename=file, directory=output_dir, text=text)
        
    print("# of files in output_dir after copy:", len(os.listdir(output_dir)), "\n\n")