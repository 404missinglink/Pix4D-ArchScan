Why do we need to fine-tune for our use case (focusing on drone/aerial survey imagery with mostly top view)? (Where it fails):

1. Fails where there are dense features in aerial imagery
2. Fails where there is dense spatial context in aerial imagery
3. Fails to understand the quantitative context as the number of features increases

Finetuning

💡Interesting question: Does VQA dataset work better for fine tuning vision language models

**FINETUNING**:

Step 1: **Baselining**

Start with instruction-tuned offspring of Pixtral in Hugging face: https://huggingface.co/Ertugrul/Pixtral-12B-Captioner-Relaxed

Already looking better than “Vanilla Pixtral”: https://huggingface.co/mistralai/Pixtral-12B-2409

Step 2: **Fine-Tune Dataset creation** 

Two-step Finetuning using :

1) Flood dataset in Image Caption setup (Already available dataset)

2) Aerial data in Image Caption setup (Captions created using Pixtral with quantitative more than descriptive questions)  (Created a small batch of 10 image-caption pairs)

##File descriptions
create_dataset.py: Creates image-text pairs for making a dataset for finetuning (Uses drone imagery and Ertugrul/Pixtral-12B-Captioner-Relaxed model from Hugging face)
finetune_LLaVa.py:Tries to finetune the Ertugrul/Pixtral-12B-Captioner-Relaxed model using targetted drone imagery-image caption dataset
Parameters are tunable.
download.py: Downloader for dataset and model from HuggingFace
login.py: Login for HuggingFace model and dataset access