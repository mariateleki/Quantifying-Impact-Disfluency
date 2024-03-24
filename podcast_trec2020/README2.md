# README2

I altered ensemble_decode_testset.py and hier_filtering_testset.py to be 1 module instead of 2. This is because I think it's easier to run the files through 1 at a time, instead of writing them all out after hier_filtering and then after ensemble_decode; I run each file through hier_filtering and then immediately after, ensemble_decode--this saves a step of having to write all the files out and deal with pickling (as they did in the original code).


1. `pip install gdown` into your `base` conda environment.
2. `gdown --id 1w8Hc4ZurjDc3O_gYKCe5EMSkKmce9kRi`
3. `tar -xvf cued_speech_at_trec2020.tar.gz`
4. Move (using `mv`) all of the models in `released_weights` to a folder named `models` in `podcast_trec2020`. May need to create the `podcast_trec2020/models` folder (use `mkdir`).

How to create the conda environment:
1. `conda create -n podcast_trec2020 python=3.7 pip ipykernel`
2. `conda activate podcast_trec2020`
3. `pip install transformers==2.11.0`
4. `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
5. `conda install pandas nltk`
6. `pip install sentencepiece==0.1.91` to downgrade the sentencepiece library per https://github.com/huggingface/transformers/issues/5001