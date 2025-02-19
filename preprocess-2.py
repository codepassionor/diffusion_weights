import os
import torch
import numpy as np
from transformers import GPT2LMHeadModel

checkpoint_dir = "/root/autodl-tmp/model_training/gpt2_training_20250113_0029"
target_layer_name = "c_fc"

param_dict = {}

checkpoints = sorted([
    d for d in os.listdir(checkpoint_dir)
    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
], key=lambda x: int(x.split('-')[1]))

for checkpoint in checkpoints:
    param_local_dict = {}
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    for name, module in model.named_modules():
        if target_layer_name in name:
            idx = name.split('.')[-3]
            param_local_dict[idx] = np.concatenate([
                module.weight.detach().cpu().numpy(),  # shape: (768, 3072)
                #module.bias.detach().cpu().numpy()[np.newaxis, :], # shape: (1, 3072)
            ], axis=0)
    param_dict[checkpoint.split("-")[-1]] = param_local_dict

out_dict = {} #重新排列字典
for i in range(1):
    i_dict = {}
    for key, value in param_dict.items():
        i_dict[key] = value[f"{i}"]
    out_dict[f"{i}"] = i_dict
    output_path = os.path.join(checkpoint_dir, "param_dict_v1.pt")
    torch.save(i_dict, output_path)

print(f"提取完成，共处理了 {len(out_dict)} 个 checkpoint。参数已保存到 {output_path}")
