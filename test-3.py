import os
import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import argparse
from my_datasets import ParaDataset_v1
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# @torch.no_grad()
# def test(args):
#     # Accelerator setup
#     accelerator = Accelerator(mixed_precision=args.mixed_precision)

#     # Define dataset and transformations
#     from torchvision import transforms
#     tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
#     dataset = ParaDataset_v1(args.data_dir)
#     test_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)

#     # Load model
#     pipeline = StableDiffusionPipeline.from_pretrained(args.output_dir)
#     pipeline.to(accelerator.device)
#     pipeline.set_progress_bar_config(disable=not accelerator.is_local_main_process)

#     # Set model to evaluation mode
#     pipeline.unet.eval()
#     mse = []
#     sim = []
#     pred = []
#     target = []

#     # Testing loop
#     os.makedirs(args.test_output_dir, exist_ok=True)
#     progress_bar = tqdm(test_dataloader, desc="Testing", disable=not accelerator.is_local_main_process)
#     for step, batch in enumerate(progress_bar):
#         idx, pixel_value, noise = batch
#         target.append(noise)

#         inputs = pixel_value.unsqueeze(0).view(1, 3, -1, 1024).to(accelerator.device)
#         inputs_latent = pipeline.vae.encode(inputs).latent_dist.sample()
#         inputs_latent = inputs_latent * pipeline.vae.config.scaling_factor

#         timesteps = torch.randint(
#             low=idx * 40 + 1,
#             high=(idx + 1) * 40,
#             size=(10,),
#             device=accelerator.device
#         )

#         text = f"Fully connected layer parameters for layer {idx} mlp of the GPT2 model"
#         text_encoded = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(accelerator.device)
#         encoder_hidden_states = pipeline.text_encoder(
#             input_ids=text_encoded['input_ids'],
#             attention_mask=text_encoded['attention_mask']
#         ).last_hidden_state

#         final_outputs = []
#         with torch.no_grad():
#             for timestep in timesteps:
#                 noisy_images = pipeline.scheduler.add_noise(
#                     inputs_latent, 
#                     torch.zeros_like(inputs_latent), 
#                     timestep
#                 )
#                 outputs = pipeline.unet(noisy_images, timestep, encoder_hidden_states=encoder_hidden_states)
#                 pred_decoded = pipeline.vae.decode(outputs.sample / pipeline.vae.config.scaling_factor).sample
#                 final_outputs.append(pred_decoded)

#         # with torch.no_grad():
#         #     output = sum(final_outputs) / len(final_outputs)
#         #     #output = output.view(1, output.size(2), -1)
#         #     noise = noise.unsqueeze(0).view(1, 3, -1, 1024).to(accelerator.device)
#         #     #noise_latent = pipeline.vae.encode(noise).latent_dist.sample()
#         #     #noise_latent = noise_latent * pipeline.vae.config.scaling_factor
#         #     #noise = noise_latent
#         #     #mse_loss = F.mse_loss(output, noise)
#         #     #outputs_normalized = F.normalize(output.view(output.size(0), -1), dim=-1)
#         #     #noise_normalized = F.normalize(noise.view(noise.size(0), -1), dim=-1)
#         #     #cosine_similarity = torch.sum(outputs_normalized * noise_normalized, dim=-1).mean()
#         #     output_mean = torch.mean(output).cpu()
#         #     noise_mean = torch.mean(noise).cpu()
#         #     pred.append(output_mean)
#         #     target.append(noise_mean)
#         #     #mse.append(mse_loss.item())
#         #     #sim.append(cosine_similarity.item())

#         # output_file = "output.txt"
#         # with open(output_file, "w") as f:
#         #     for sim_val, mse_val in zip(sim, mse):
#         #         f.write(f"sim: {sim_val}, mse: {mse_val}\n")
    
#     plt.figure(figsize=(8, 6))
#     plt.plot(pred, label='Prediction', marker='o')
#     plt.plot(target, label='Target', marker='x')
    
#     plt.title('Prediction vs Target', fontsize=16)
#     plt.xlabel('Index', fontsize=12)
#     plt.ylabel('Value', fontsize=12)
#     plt.legend(fontsize=12)
    
#     plt.savefig('figure.png', dpi=300)
#     plt.close()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Stable Diffusion Testing")
#     parser.add_argument("--output_dir", type=str, default="./stable-diffusion-2-1-finetuned", help="Path to the finetuned model")
#     parser.add_argument("--test_output_dir", type=str, default="./test_results", help="Directory to save test results")
#     parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size")
#     parser.add_argument("--image_size", type=int, default=512, help="Input image size")
#     parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision mode")
#     parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/llm_weight/param_dict_v1.pt", help="Path to the test data")
#     args = parser.parse_args()

#     test(args)







import numpy as np

# @torch.no_grad()
# def test(args):
#     # Accelerator setup
#     accelerator = Accelerator(mixed_precision=args.mixed_precision)

#     # Define dataset and transformations
#     from torchvision import transforms
#     tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
#     dataset = ParaDataset_v1(args.data_dir)
#     test_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)

#     # Load model
#     pipeline = StableDiffusionPipeline.from_pretrained(args.output_dir)
#     pipeline.to(accelerator.device)
#     pipeline.set_progress_bar_config(disable=not accelerator.is_local_main_process)

#     # Set model to evaluation mode
#     pipeline.unet.eval()
#     pred = []
#     target = []
#     final_outputs = []
#     # Testing loop
#     os.makedirs(args.test_output_dir, exist_ok=True)
#     progress_bar = tqdm(test_dataloader, desc="Testing", disable=not accelerator.is_local_main_process)
#     for step, batch in enumerate(progress_bar):
#         idx, pixel_value, noise = batch
#         target.append(noise.to(accelerator.device))

#         inputs = pixel_value.unsqueeze(0).view(1, 3, -1, 1024).to(accelerator.device)
#         inputs_latent = pipeline.vae.encode(inputs).latent_dist.sample()
#         inputs_latent = inputs_latent * pipeline.vae.config.scaling_factor

#         timesteps = torch.randint(
#             low=idx * 40 + 1,
#             high=(idx + 1) * 40,
#             size=(10,),
#             device=accelerator.device
#         )

#         text = f"Fully connected layer parameters for layer {idx} mlp of the GPT2 model"
#         text_encoded = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(accelerator.device)
#         encoder_hidden_states = pipeline.text_encoder(
#             input_ids=text_encoded['input_ids'],
#             attention_mask=text_encoded['attention_mask']
#         ).last_hidden_state

#         with torch.no_grad():
#             pred = []
#             for timestep in timesteps:
#                 '''
#                 noisy_images = pipeline.scheduler.add_noise(
#                     inputs_latent, 
#                     torch.zeros_like(inputs_latent), 
#                     timestep
#                 )
#                 '''
#                 outputs = pipeline.unet(inputs_latent, timestep, encoder_hidden_states=encoder_hidden_states)
#                 pred_decoded = pipeline.vae.decode(outputs.sample / pipeline.vae.config.scaling_factor).sample
#                 pred.append(pred_decoded)
#             final_output = sum(pred) / len(pred)
#             final_outputs.append(final_output)
import os
import torch
from tqdm import tqdm
from accelerate import Accelerator
from transformers import CLIPTokenizer
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline
from my_datasets import ParaDataset_v1  # Assuming this is your dataset class

@torch.no_grad()
def test(args):
    # Accelerator setup
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    # Define dataset and transformations
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    dataset = ParaDataset_v1(args.data_dir)
    test_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False)

    # Load model
    pipeline = StableDiffusionPipeline.from_pretrained(args.output_dir)
    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=not accelerator.is_local_main_process)

    # Set model to evaluation mode
    pipeline.unet.eval()
    pred = []
    target = []
    final_outputs = []

    # Create the output directory if it doesn't exist
    os.makedirs(args.test_output_dir, exist_ok=True)
    
    # Testing loop
    progress_bar = tqdm(test_dataloader, desc="Testing", disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(progress_bar):
        idx, pixel_value, noise = batch
        target.append(noise.to(accelerator.device))

        inputs = pixel_value.unsqueeze(0).view(1, 3, -1, 1024).to(accelerator.device)
        inputs_latent = pipeline.vae.encode(inputs).latent_dist.sample()
        inputs_latent = inputs_latent * pipeline.vae.config.scaling_factor

        timesteps = torch.randint(
            low=idx * 40 + 1,
            high=(idx + 1) * 40,
            size=(10,),
            device=accelerator.device
        )

        text = f"Fully connected layer parameters for layer {idx} mlp of the GPT2 model"
        text_encoded = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(accelerator.device)
        encoder_hidden_states = pipeline.text_encoder(
            input_ids=text_encoded['input_ids'],
            attention_mask=text_encoded['attention_mask']
        ).last_hidden_state

        pred = []
        for timestep in timesteps:
            outputs = pipeline.unet(inputs_latent, timestep, encoder_hidden_states=encoder_hidden_states)
            pred_decoded = pipeline.vae.decode(outputs.sample / pipeline.vae.config.scaling_factor).sample
            pred.append(pred_decoded)
        
        final_output = sum(pred) / len(pred)
        final_outputs.append(final_output)

        # Save predicted results after each checkpoint
        checkpoint_dir = os.path.join(args.test_output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(final_output.shape)
        # Save the model's predictions as a safetensors file
        torch.save(final_output, os.path.join(checkpoint_dir, "model.safetensors"))


    # all_noises = []
    # all_outputs = []
    # mse = []
    # sim = []
    # for output, noise_tensor in zip(final_outputs, target):
    #     all_outputs.append(output.cpu().numpy().flatten())
    #     all_noises.append(noise_tensor.cpu().numpy().flatten())
    #     output = output.view(1, output.size(2), -1)
    #     mse_loss = F.mse_loss(output, noise_tensor)
    #     outputs_normalized = F.normalize(output.view(output.size(0), -1), dim=-1)
    #     noise_normalized = F.normalize(noise_tensor.view(noise.size(0), -1), dim=-1)
    #     cosine_similarity = torch.sum(outputs_normalized * noise_normalized, dim=-1).mean()
    #     mse.append(mse_loss)
    #     sim.append(cosine_similarity)

    # output_file = "output.txt"
    # with open(output_file, "w") as f:
    #     for sim_val, mse_val in zip(sim, mse):
    #         f.write(f"sim: {sim_val}, mse: {mse_val}\n")

    # markers = ['o', 's']  # Circle for predicted, square for actual
    # colors = ['blue', 'red']  # Blue for predicted, red for actual
    
    # plt.figure(figsize=(25, 8))
    
    # Plot all outputs and noises for each epoch

    # # Iterate through each epoch and corresponding data
    # for epoch_idx, (output_flat, noise_flat) in tqdm(enumerate(zip(all_outputs, all_noises)), 
    #                                                  total=len(all_outputs), 
    #                                                  desc="Plotting Epochs"):
    #     # Create a new figure for each epoch
    #     plt.figure(figsize=(10, 6))
        
    #     # Plot predicted parameters
    #     plt.plot(
    #         output_flat,
    #         label=f'Predicted (Epoch {epoch_idx + 1})',
    #         color=colors[0],
    #         alpha=0.7,
    #         linestyle='-',
    #         marker=markers[0]
    #     )
        
    #     # Plot actual parameters
    #     plt.plot(
    #         noise_flat,
    #         label=f'Actual (Epoch {epoch_idx + 1})',
    #         color=colors[1],
    #         alpha=0.7,
    #         linestyle='--',
    #         marker=markers[1]
    #     )
        
    #     # Set title and labels
    #     plt.title(f'Parameter Distribution for Epoch {epoch_idx + 1}', fontsize=16)
    #     plt.xlabel('Parameter Index', fontsize=12)
    #     plt.ylabel('Parameter Value', fontsize=12)
        
    #     # Add legend and grid
    #     plt.legend(fontsize=10, loc='upper right')
    #     plt.grid(True)
        
    #     # Save the plot for this epoch
    #     plt.savefig(f"{args.test_output_dir}/epoch_{epoch_idx + 1}.png", dpi=300)
        
    #     # Close the figure to free memory
    #     plt.close()
    # 用于存储所有 epoch 的降维数据
    # all_pca_predicted = []
    # all_pca_actual = []
    # epoch_labels = []
    
    # # Perform PCA and collect data for each epoch
    # for epoch_idx, (output_flat, noise_flat) in tqdm(enumerate(zip(all_outputs, all_noises)), 
    #                                                  total=len(all_outputs), 
    #                                                  desc="Performing PCA and Collecting Data"):
    #     # Perform PCA to reduce both predicted and actual parameters to 2D
    #     pca = PCA(n_components=2)
        
    #     # Combine the data to fit PCA together, then split back to predicted and actual
    #     predicted_results = PCA(n_components=2).fit_transform(output_flat.reshape(-1, 1))

    #     actual_results =  PCA(n_components=2).fit_transform(noise_flat.reshape(-1, 1))       
        
    #     # Collect the results
    #     all_pca_predicted.append(predicted_results)
    #     all_pca_actual.append(actual_results)
    
    # # Flatten the list of arrays into a single array
    # all_pca_predicted = np.vstack(all_pca_predicted)
    # all_pca_actual = np.vstack(all_pca_actual)
   
    # # Create a scatter plot for all epochs
    # plt.figure(figsize=(10, 8))
    
    # # Plot all predicted points
    # plt.scatter(
    #     all_pca_predicted[:, 0], 
    #     all_pca_predicted[:, 1], 
    #     label='Predicted', 
    #     color='blue', 
    #     alpha=0.5, 
    #     marker='o'
    # )
    
    # # Plot all actual points
    # plt.scatter(
    #     all_pca_actual[:, 0], 
    #     all_pca_actual[:, 1], 
    #     label='Actual', 
    #     color='red', 
    #     alpha=0.5, 
    #     marker='x'
    # )
    
    # # Set title and labels
    # plt.title('2D PCA Comparison Across All Epochs', fontsize=16)
    # plt.xlabel('PCA Component 1', fontsize=12)
    # plt.ylabel('PCA Component 2', fontsize=12)
    
    # # Add legend and grid
    # plt.legend(fontsize=12, loc='upper right')
    # plt.grid(True)
    
    # # Save the combined plot
    # plt.savefig(f"{args.test_output_dir}/combined_pca_comparison.png", dpi=300)
    
    # # Display the plot (optional)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Testing")
    parser.add_argument("--output_dir", type=str, default="./stable-diffusion-2-1-finetuned", help="Path to the finetuned model")
    parser.add_argument("--test_output_dir", type=str, default="./test_results", help="Directory to save test results")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision mode")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/llm_weight/param_dict_v1.pt", help="Path to the test data")
    args = parser.parse_args()

    test(args)



