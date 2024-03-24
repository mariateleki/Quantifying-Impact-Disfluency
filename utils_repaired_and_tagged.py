import os
import re

import utils_general

def get_repaired_or_tagged_transcript(transcript, mode):
    
    sentences = transcript.split("\n")
    
    new_transcript = []
    for sentence in sentences:
        
        words = sentence.split()
        
        new_sentence = []
        for i in range(0,len(words)-1,2):
            
            # perform some basic cleaning on transcripts coming out of the disfluency annotation model
            w = words[i]
            if i == 0:  # capitalize words at the beginning of sentences
                w = w.capitalize()
            if i == len(words)-2:  # add periods to the end of sentences
                w = w + "."
            if w == "i":  # change lowercase "i" to uppercase "I"
                w = "I"
            
            if words[i+1] == "E":
                
                if mode == "repaired":  # do not add these words to the new transcript, bc they are marked as disfluent
                    pass
                elif mode == "tagged":  # add these words to the new transcript with tags, bc they are disfluent
                    
                    # this word needs to be capitalized if it's now the first word of the sentence
                    if len(new_sentence) == 0:
                        new_sentence.append("<DIS> " + w.capitalize() + " <\DIS>")
                    else:
                        new_sentence.append("<DIS> " + w + " <\DIS>")
                
            elif words[i+1] == "_":
                # do add these words to the new transcript, bc they are marked as fluent
                
                # this word needs to be capitalized if it's now the first word of the sentence
                if len(new_sentence) == 0:
                    new_sentence.append(w.capitalize())
                else:
                    new_sentence.append(w)
         
        new_transcript.append(" ".join(new_sentence))
        
    # convert the new_transcript list of words into a new_transcript string
    new_transcript = " ".join(new_transcript)

    # additional cleaning to join contractions back together
    new_transcript = new_transcript.replace(" n't", "n't")
    new_transcript = re.sub(r"(\w+) (')", r"\1\2", new_transcript)
    
    return new_transcript

def write_repaired_and_tagged_transcripts(input_dir="spotify_test", seed_number=0):
    
    if input_dir == "spotify_test":
        seed_number = -1  # doesn't matter
        annotated_dir = os.path.join(utils_general.PATH_TO_PROJECT, input_dir, "annotated")
        repaired_dir = os.path.join(utils_general.PATH_TO_PROJECT, input_dir, "-1")
        tagged_dir = os.path.join(utils_general.PATH_TO_PROJECT, input_dir, "tagged")
        
    elif input_dir == "spotify_train":
        annotated_dir = os.path.join(utils_general.PATH_TO_PROJECT, input_dir, "annotated", f"percent_0.01_random_state_{seed_number}")
        repaired_dir = os.path.join(utils_general.PATH_TO_PROJECT, input_dir, f"percent_0.01_random_state_{seed_number}", "-1")
        tagged_dir = os.path.join(utils_general.PATH_TO_PROJECT, input_dir, f"percent_0.01_random_state_{seed_number}", "tagged")

    utils_general.create_and_or_clear_this_dir(repaired_dir)
    utils_general.create_and_or_clear_this_dir(tagged_dir)

    files_list = [f for f in os.listdir(annotated_dir) if (not f.endswith("_orig_dys.txt") and f.endswith("_dys.txt") and not f.endswith("-checkpoint.txt"))]
    for f in files_list:

        transcript = utils_general.read_file(os.path.join(annotated_dir,f))

        repaired_transcript = ""
        repaired_transcript = get_repaired_or_tagged_transcript(transcript, "repaired")
        utils_general.write_file(f.replace("_dys",""), repaired_dir, repaired_transcript)

        tagged_transcript = ""
        tagged_transcript = get_repaired_or_tagged_transcript(transcript, "tagged")
        utils_general.write_file(f.replace("_dys",""), tagged_dir, tagged_transcript)