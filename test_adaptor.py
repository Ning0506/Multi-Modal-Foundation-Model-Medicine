import ast
import torch

from PIL import Image
import numpy as np

import cv2
import numpy as np
'''
path = 's50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg'
img = Image.open(path)
img = img.convert('RGB')
img = img.resize((224, 224))

img = np.array(img)/255.

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


img = img - imagenet_mean
img = img / imagenet_std
'''
import ast

# Open the .txt file and read its contents

import fire

from llama import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 1,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    prompts = data_array
    print('prompt type', type(prompts))
    '''[
        # For these prompts, the expected answer is the natural continuation of the prompt
        img,
        
    ]'''
    
    results = generator.classification_mission(
        prompts,
    )
    for prompt, result in zip(prompts, results):
#         print(prompt)
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    with open('latent_representation.txt', 'r') as file:
        data_str = file.read()

    # Convert the string representation to a list of lists
    data_list = ast.literal_eval(data_str)


    data_array = np.array(data_list)
    fire.Fire(main)