# speedrun

to set up this repo do as such:

1. clone it and run setup.sh
2. generate data by cd data then bash process_data.sh
3. login to wandb using wandb login and your key (if you have no account, make one; it's free)
4. train with bash run.sh (you may want to nohup this in case of crashing)

the repo is structured into two main parts: alg and data (folders). within alg, the main training logic lies, including two optimizers, which you can modify. 

The main training arguments are in alg/args.py, and the the loss arguments are in objectives (loss.py, objectives.py are the core logic). 

the data folder contains the data logic. 

The main place for modification for data is in data/configs/data.json, which has a json config of huggingface datasets that is tokenized and then saved to disk (for ~1 billion tokens). each dataset has an associated fraction of the 1 billion tokens assigned to it (e.g. semran1/synth-cc is assigned 22% of the total mix). Note that if the data has any errors tokenizing, a common fix is simply to decreas the number of workers in data/scripts/download_and_tokenize.py.

for evaluation, run bash eval.sh

A baseline run (achieved with exactly as the settings are in the repo right now achieved)
mmlu (0 shot), acc: 44.08
mmlu (5 shot), acc: 46.30
hellaswag (0 shot), acc norm: 51.99
arc_challenge (0 shot), acc norm: 36.00
