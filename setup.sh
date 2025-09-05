conda create -n speedrun python=3.10
conda activate speedrun
pip install torch==2.6.0 packaging wheel ninja
pip install flash-attn==2.7.4.post1
pip install -r requirements.txt