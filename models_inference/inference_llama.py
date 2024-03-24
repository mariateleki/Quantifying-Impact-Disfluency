# Make sure to go through and update this script with your own data (i.e., your_access_token, etc.). You may also have to adjust the folder names/places.

import os
from transformers import LlamaForCausalLM, AutoTokenizer,AutoModelForCausalLM
import torch
import os
import glob
import pandas as pd
from tqdm import tqdm

model_name = "Llama-2-7b-chat-hf"

access_token = [your_access_token]
tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model_name}", use_auth_token=access_token)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM.from_pretrained(f"meta-llama/{model_name}", use_auth_token=access_token, device_map="auto")

def summarize(folder_path, model_name):
    text_files = glob.glob(os.path.join(folder_path, "*.txt"))

    for file_path in tqdm(text_files):

        df = pd.read_csv(file_path, sep='\t', header=None, names=['text'])
        sentence = df['text'].to_list()[0]
        prompt = f"[INST]summarize: {sentence}[/INST]"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        outputs = model.generate(inputs.input_ids, min_new_tokens=56, max_new_tokens=144)

        generated_summary = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        output_sentence = generated_summary.split("[/INST]")[1].strip()

        file_path_split = os.path.basename(file_path).split(".")
        assert len(file_path_split) == 2
        save_dir = f"{model_name}-summary/{folder_name}"
        save_file_name = f"{file_path_split[0]}-summary.txt"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, save_file_name), "w") as file:
            file.write(output_sentence)


for folder_name in ["0", "interjections", "false-starts", "repeats-and-false-starts", "repeats-and-interjections", "interjections-and-false-starts", "all-3"]:

    folder_path = [your_folder_path]
    model_name = "llama2-7b-chat"
    summarize(folder_path, model_name)