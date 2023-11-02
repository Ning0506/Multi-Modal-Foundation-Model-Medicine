# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

import numpy as np
from torch import nn

from llama.mae import *

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."




class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        
        

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.visual_proj = nn.Linear(768, 4096)
        self.visual_proj_norm = nn.LayerNorm(4096)
        #self.classification_head = nn.Linear(embed_dim, num_classes)
        
        self.mae_vit = prepare_model('llama/mae_pretrain_vit_base.pth', 'mae_vit_base_patch16')
        
        self.classification_head = nn.Linear(4096, 14)
        
#         for name, p in self.mae_vit.named_parameters():
#             print(name, p.dtype)
#             break
        
    def visual_forward_single(self, img):
#         x = torch.tensor(img)

        # make it a batch-like
#         x = x.unsqueeze(dim=0)
#         x = torch.einsum('nhwc->nchw', x)
#         print(x.float().dtype)
        
#         x = x.type(torch.float16)
        print(img.shape)
        latent = self.mae_vit(img)

        # run MAE
#         latent = self.mae_vit(x.float())

        return latent

    @torch.inference_mode()
    def classification_mission(self,
                              image):
        params = self.model.params
        print(type(image))
        print(image.shape)
        
        #生成图像embedding
        #image_embedding = self.visual_forward_single(image)
        image_embedding = image
        visual_query = torch.tensor(image_embedding)
        visual_query = visual_query.to(self.visual_proj.weight.dtype)
        print(visual_query.size())
        #visual_query = visual_query.squeeze(0)
#         print(visual_query.size())
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)
#         print(visual_query.size())
        #visual_query = self.forward_visual(imgs)
        # visual_query 的size是 1*hyper*4096
        # 下一步需要把他放进模型的第一层，跳过embedding层
        
        logits = self.model.forward(visual_query, 0)
        print('logits dimension', logits.size())
        
        CLS = logits[:, 0]
        
        CLS = CLS.to(self.classification_head.weight.dtype)
        print('CLS dimension', CLS.size())
        
        logits = self.classification_head(CLS)
        
        print('final output dimension', logits.size())
        return logits
    

    