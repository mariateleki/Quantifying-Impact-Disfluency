# README.md

# Datasets

## Spotify
The Spotify Podcasts Dataset can be obtained by applying at [this link](https://podcastsdataset.byspotify.com/), as access to the datset is controlled by Spotify. 

`spotify` contains the transcripts and their transformations.

All paths in the script are based off of this location in the larger Spotify-Podcasts dataset:
`Spotify-Podcasts/EN/podcasts-no-audio-13GB`

Copy `podcasts-no-audio-13GB` into a folder named `podcasts-no-audio-13GB` locally.

# Running this code
1. Go to utils_general.py and edit the PATHS at the top of the file. 
2. Create the Spotify dataset in this project by running its notebook (Spotify.ipynb).

# Data Transformations
There are 2 types of transformations that we do on this data: 
1. **Synthetic Disfluency Augmentation (0 to 10 transformations)**: We run our repeats, interjections, and false starts transformations on the podcast transcripts.
2. **Disfluency Annotation Model (-1 transformation)**: We run Jamshid Lou and Johnson's (2020) model on the 0 podcast transcripts and remove words marked as disfluent. 
    * The repository for Jamshid Lou and Johnson (2020) is [here](https://github.com/pariajm/joint-disfluency-detector-and-parser). On this page, they note [here](https://github.com/pariajm/joint-disfluency-detector-and-parser#using-the-trained-models-for-disfluency-tagging) to use [this](https://github.com/pariajm/english-fisher-annotations) repository for "us\[ing\] the trained models to disfluency label your own data."
    * Steps to run it: 
        1. Use [these instructions](https://github.com/pariajm/english-fisher-annotations#using-the-model-to-annotate-fisher) to (1) clone the repo, (2) download the model and necessary resources for the model, and (3) unzip the model.
        2. Create a virtual environment which meets [these requirements](https://github.com/pariajm/english-fisher-annotations#software-requirements) to run the script.
        3. Run the script [using these instructions](https://github.com/pariajm/english-fisher-annotations#using-the-model-to-annotate-fisher) on these files:
            * `mkdir spotify_test/annotated`
            * `cd english-fisher-annotations`
            * `python main.py --input-path ../spotify_test/0 --output-path ../spotify_test/annotated --model ./model/swbd_fisher_bert_Edev.0.9078.pt`

# Notes
* `pip install sentencepiece google protobuf` to get T5 working right now, otherwise you get errors
* `conda install evaluate` and `pip install rouge_score` to get calculate_rouge_scores.py working


