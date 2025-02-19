import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import os
from tqdm import tqdm

def load_checkpoint(ckpt_path):
    model = GPT2LMHeadModel.from_pretrained(ckpt_path)
    return model

def merge_weights(w1, w2, alpha):
    return (1 - alpha) * w1 + alpha * w2

def evaluate_model(model, tokenizer, dataset):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        for sample in tqdm(dataset):
            input_ids = tokenizer(
                sample['text'],
                return_tensors='pt',
                truncation=True,
                max_length=1024,  
                padding=False
            ).input_ids.to(device)
            
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item() * input_ids.size(1) 
            
            shifted_input_ids = input_ids[:, 1:] 
            shifted_logits = logits[:, :-1, :] 
            predictions = torch.argmax(shifted_logits, dim=-1)
            correct += (predictions == shifted_input_ids).sum().item()
            total += shifted_input_ids.numel()
    
    ppl = torch.exp(torch.tensor(total_loss / total)).item()
    acc = correct / total
    return ppl, acc

if __name__ == "__main__":
    ckpt1_path = "/root/autodl-tmp/llm_weight/test_results/checkpoint-16/model.safetensors"
    ckpt2_path = "/root/autodl-tmp/llm_weight/test_results/checkpoint-19/model.safetensors"
    base_ckpt_path = "/root/autodl-tmp/model_training/gpt2_training_20250113_0029/gpt2"
    
    model1 = torch.load(ckpt1_path)
    model2 = torch.load(ckpt2_path)
    
    weight1 = model1.squeeze(0).reshape(768, 3 * 1024)
    weight2 = model2.squeeze(0).reshape(768, 3 * 1024)
    
    tokenizer = GPT2Tokenizer.from_pretrained("/root/autodl-tmp/model_training/gpt2_training_20250113_0029/gpt2/")
    dataset = load_dataset('json', data_files='/root/autodl-tmp/lambada/data/lambada_test.jsonl')
    dataset = dataset['train']
    
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8]:
        #merged_weight = merge_weights(weight1, weight2, alpha)
        merged_weight = weight2
        model_merged = load_checkpoint(base_ckpt_path)
        model_merged.state_dict()['transformer.h.0.mlp.c_fc.weight'].copy_(merged_weight)
        
        ppl, acc = evaluate_model(model_merged, tokenizer, dataset)
        #print(f"Alpha: {alpha} | PPL: {ppl:.4f} | ACC: {acc:.4f}")
        with open("/root/autodl-tmp/model_training/model_merge_results.txt", "a") as f:
            f.write(f"Alpha: {alpha} | PPL: {ppl:.4f} | ACC: {acc:.4f}\n")