import numpy as np
import random
import re
import os
from nltk.tokenize import sent_tokenize

from collections import Counter

import utils_general

def capitalize_after_period(text):
        return re.sub(r'(?<=\. )\w', lambda m: m.group().upper(), text)

# util function used to create randomly-sized sublists for various transformations
def get_sublists(transcript_text, rng):
    
    # create sublists of random size (according to params below) of the transcript
    text_list = transcript_text.split()
    sublists = []
    start_index = 0
    while start_index < len(text_list):

        # get a random sublist
        mu = 10
        sigma = 1
        rand_int = int(rng.normal(mu, sigma)) 
        
        # move the end_index of the list up by that rand_int number of spots
        end_index = start_index + rand_int

        # add it to the big list of random sublists 
        sublist = text_list[start_index:end_index]  # if end_index is too big, python just takes the last index
        sublists.append(sublist)

        # then move start_index up
        start_index = end_index
        
    return sublists

def get_transformed_text(N, transcript_text, list_to_use, rng):

    # get sublists of random size of the transcript text
    sublists = get_sublists(transcript_text, rng)

    substrings = []
    for i in range(len(sublists)):
        sublist = sublists[i]
        
        if i+1 < len(sublists)-1:
            next_sublist = sublists[i+1]
        else:
            next_sublist = None
        
        shift_period = False
        for j in range(0,N):
            
            # if this is a repeats transformation
            if list_to_use == []:
                word = sublist[-1]
                word = word.replace(".","").replace("!","").replace("?","")
            
            # if this is an interjections transformation
            else:
                word = rng.choice(list_to_use)
            
            # if this is the end of the transcript, do NOT append anything
            if next_sublist == None:
                continue
            
            # if there is 1 item being appended, and a period needs to be shifted
            elif (j == 0) and ("." in sublist[-1]) and (N == 1):
                last_word = sublist.pop().replace(".","").replace("!","").replace("?","")
                sublist.append(last_word)
                sublist.append(word + ".")
                
            # if there is >1 item being appended, and a period needs to be shifted
            elif (j == 0) and ("." in sublist[-1]):
                last_word = sublist.pop().replace(".","").replace("!","").replace("?","")
                sublist.append(last_word)
                sublist.append(word)
                shift_period = True
            
            # if there is >1 item being appended, and this is the last item, so it's time to append that period back in
            elif (next_sublist) and (j == N-1) and (shift_period == True):
                sublist.append(word + ".")
                
            # if there's nothing special that needs to happen, just append that word
            else:
                sublist.append(word)

        substring = " ".join(sublist)
        substrings.append(substring)

    new_text = " ".join(substrings)
    
    new_text = capitalize_after_period(new_text)
    
    if new_text[-1] != ".":
        new_text = new_text + "."
    
    return new_text

def get_repeats_text(n, transcript_text, rng):
    return get_transformed_text(N=n, transcript_text=transcript_text, list_to_use=[], rng=rng)

def get_interjections_text(n, transcript_text, rng):
    return get_transformed_text(N=n, transcript_text=transcript_text, list_to_use=["uh", "um", "well", "like", "so", "okay", "you know", "I mean"], rng=rng)

def get_false_starts_text(n, transcript_text, rng):
    return get_transformed_text_2(N=n, transcript_text=transcript_text, rng=rng)

# does the false starts only transformation on the transcript_text
def get_transformed_text_2(N, transcript_text, rng):
    
    # break the transcript_text into sentences
    sentences = sent_tokenize(transcript_text)
    
    # and prepare to build the list of new sentences
    new_sentences = []
    
    # get the sentences with len >= 4
    sentences_len_gteq4 = [s for s in sentences if len(s.split(" ")) >= 4]
    
    # randomly determine which sentences to inject a false start into 
    sentences_len_gteq4_mask = rng.choice(1+1,size=len(sentences_len_gteq4), replace=True,p=[0.80,0.20])
    sentences_len_gteq4_mask_index = 0
    
    # then iterate through the sentences and inject those false starts based on the mask
    for i in range(len(sentences)):
        
        # get the current sentence
        sentence = sentences[i]
        #print(sentence)
        
        # and then determine whether or not to inject a false start into that sentence
        if len(sentence.split(" ")) >= 4:
            
            # if the sentence gets a false start
            if sentences_len_gteq4_mask[sentences_len_gteq4_mask_index] == 1:
                
                try:
                    
                    sentence_list = []
                    for word in sentence.split(" "):
                        sentence_list.append(word)
                    
                    subsentence_list = sentence_list[0:2]
                    rest_of_sentence_list = sentence_list[2:]
                    
                    # subsentence_list = [s.replace(".","") for s in subsentence_list]
                    subsentence = " ".join(subsentence_list).strip()
                    rest_of_sentence = " ".join(rest_of_sentence_list).strip()

                    # create a lowercased version of the subsentence so it can be appended multiple times
                    lowercased_subsentence = ""
                    if (not subsentence.startswith("I ")) and (not subsentence.startswith("I'")):
                        lowercased_subsentence = subsentence[0].lower() + subsentence[1:].strip()
                    else:
                        lowercased_subsentence = subsentence

                    # build the new sentence
                    new_sentence = ""
                    new_sentence += subsentence + " "  # append the regularly captialized version
                    for _ in range(0,N):
                        new_sentence += lowercased_subsentence + " "  # then, append the rest as lowercased versions   
                    
                    new_sentence += rest_of_sentence  # then, append the rest of the sentence!
                    new_sentences.append(new_sentence)
                
                except:
                    print(subsentence_list)
            
            # if the sentence does not get a false start
            else:
                
                # just append the original sentence
                new_sentences.append(sentence)
            
            # whether it gets appended or not, it's still greater than length 4, so the index counter gets advanced by 1
            sentences_len_gteq4_mask_index += 1
        
        # if the sentence is less than length 4, the original sentence gets appended
        else:
            new_sentences.append(sentence)

    # join all the sentences to build the new transcript
    new_text = " ".join(new_sentences)
    
    new_text = capitalize_after_period(new_text)
    
    if new_text[-1] != ".":
        new_text = new_text + "."
    
    # and return the new transcript
    return new_text

def get_num_repeats(transcript_text):
    words = transcript_text.strip().split()

    count = 0
    for i in range(len(words)-1):
        
        # get the words
        word = words[i]
        next_word = words[i+1]
        
        # clean the words
        word = word.lower().replace("?","").replace("!","").replace(".","").replace(",","")
        next_word = next_word.lower().replace("?","").replace("!","").replace(".","").replace(",","")
        
        if word == next_word:
            count += 1
            
    return count

def get_num_interjections(transcript_text):
    one_word_interjections = ["uh", "um", "well", "like", "so", "okay"]
    two_word_interjections = ["you know", "I mean"]
    
    count = 0
    words = transcript_text.split(" ")
    for i in range(len(words)-1):
        
        # get the words
        word = words[i]
        next_word = words[i+1]
        
        # clean the words
        word = word.lower().replace("?","").replace("!","").replace(".","").replace(",","")
        next_word = next_word.lower().replace("?","").replace("!","").replace(".","").replace(",","")
        
        if word in one_word_interjections:
            count += 1
        
        for i in two_word_interjections:
            
            i = i.lower().replace("?","").replace("!","").replace(".","").replace(",","")
            
            if (word == i.split(" ")[0]) and (next_word == i.split(" ")[1]):
                count +=1
            
    return count

def get_num_false_starts(transcript_text):
    count = 0
    
    # break the transcript_text into sentences
    sentences = sent_tokenize(transcript_text)
    
    # then iterate through the sentences and inject those false starts
    for sentence in sentences:
        
        if len(sentence.split(" ")) >= 4:  # otherwise the sentence is too short to conform to our definition of false_starts

            # get the subsentence (this is the part that will be repeated howevermany times)
            subsentence_list = sentence.split(" ")[0:2]  # if sentence is too short, python will just take as much length as it can
            first_subsentence = " ".join(subsentence_list).strip().lower().replace("?","").replace("!","").replace(".","").replace(",","")

            subsentence_list = sentence.split(" ")[2:4]
            second_subsentence = " ".join(subsentence_list).strip().lower().replace("?","").replace("!","").replace(".","").replace(",","")

            # # create a lowercased version of the subsentences so they can be compared
            # first_subsentence = first_subsentence[0].lower() + first_subsentence[1:].strip()
            # second_subsentence = second_subsentence[0].lower() + second_subsentence[1:].strip()

            # if they're the same, count them!
            if first_subsentence == second_subsentence:
                count += 1
    
    return count

def write_disfluency_counts_to_csv(input_dir):
    
    # create the csv dir if it doesn't exist
    utils_general.just_create_this_dir(os.path.join(utils_general.PATH_TO_PROJECT, "csv"))

    # write out the first (header) line of the csv
    output_file_name = input_dir.split("/")[-1] + "__" + "disfluency-reporting" + ".csv"
    output_path = os.path.join(os.path.join(utils_general.PATH_TO_PROJECT, "csv"), output_file_name)
    
    with open(output_path, mode="w") as f:
        f.write("episode_id,num_repeats,num_interjections,num_false_starts\n")

    # iterate through all the files in the current directory
    list_of_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    for file in list_of_files:

        # get the transcript text
        with open(os.path.join(input_dir,file)) as f:
            transcript_text = f.read()

        # get each part that needs to be written out to the csv line
        episode_id = file.replace(".txt", "")
        
        num_repeats = get_num_repeats(transcript_text)
        num_interjections = get_num_interjections(transcript_text)
        num_false_starts = get_num_false_starts(transcript_text)

        # write the line out to file
        with open(output_path, mode="a") as f:
            f.write("{},{},{},{}\n".format(episode_id, num_repeats, num_interjections, num_false_starts))
                

