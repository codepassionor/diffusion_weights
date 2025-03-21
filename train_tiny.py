import os
import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL
from transformers import AutoTokenizer, get_scheduler, CLIPTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import argparse
from my_datasets import ParaDataset

def fill(inputs):
    return inputs.repeat(1, 384, 32)
def recover(inputs):
    return inputs[:, :2, :8]

def main(args):
    # Accelerator setup
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    # Define dataset and transformations
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    dataset = ParaDataset(args.data_dir)
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # Load model and scheduler
    pipeline = StableDiffusionPipeline.from_pretrained(args.model_name)
    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    vae = AutoencoderKL.from_pretrained("/root/autodl-tmp/llm_weight/fine_tuned_vae_last")
    vae.to(accelerator.device)
    pipeline.vae = vae

    # Freeze all weights except text encoder and U-Net
    pipeline.text_encoder.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.unet.train()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_dataloader) * args.num_epochs
    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    # Prepare for training
    pipeline, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipeline, optimizer, train_dataloader, lr_scheduler
    )
    pipeline.to(accelerator.device)
    # Training loop
    for epoch in range(args.num_epochs):
        pipeline.unet.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(progress_bar):
            idx, pixel_value, targets = batch
            for layer_tag in pixel_value:
                
                inputs = fill(pixel_value[layer_tag])
                target = fill(targets[layer_tag])
                with torch.no_grad():
                    inputs = inputs.unsqueeze(1).expand(1, 3, -1, -1)
                    target = target.unsqueeze(1).expand(1, 3, -1, -1)
                    inputs_latent = pipeline.vae.encode(inputs).latent_dist.sample()
                    inputs_latent = inputs_latent * pipeline.vae.config.scaling_factor

                #timesteps = torch.randint(idx * 40 + 1, (idx + 1) * 40, (1,), device=accelerator.device)
                #noisy_images = scheduler.add_noise(inputs_latent, noise_latent, timesteps)
                timesteps = torch.tensor([idx * 40], device=accelerator.device)
                
                text = f"Fully connected layer parameters for layer {layer_tag} mlp of the GPT2 tiny model"
                text_encoded = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(accelerator.device)
                encoder_hidden_states = pipeline.text_encoder(
                    input_ids=text_encoded['input_ids'],
                    attention_mask=text_encoded['attention_mask']
                ).last_hidden_state
                outputs = pipeline.unet(inputs_latent, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                
                pred_decoded = pipeline.vae.decode(outputs / pipeline.vae.config.scaling_factor).sample
                loss = torch.nn.functional.mse_loss(pred_decoded, target)

                # Backpropagation
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % args.logging_steps == 0:
                progress_bar.set_postfix({"loss": loss.item()})

            if step % args.save_steps == 0 and accelerator.is_local_main_process:
                pipeline.save_pretrained(args.output_dir)  
            
        # Save model after each epoch
    if accelerator.is_local_main_process:
        pipeline.save_pretrained(args.output_dir)

    print("Training complete. Model saved at:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Training")
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-2-1-base", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./stable-diffusion-2-1-finetuned-tiny", help="Directory to save the model")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps for learning rate scheduler")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision training")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=5000, help="Steps to save the model")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/llm_weight/param_data/param_dict_gpt2_tiny.pt", help="Path to the directory containing images")
    args = parser.parse_args()

    main(args)