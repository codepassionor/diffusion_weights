import torch
import os
import numpy as np
from transformers import GPT2Model, logging
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Reduce verbosity of transformers logging
logging.set_verbosity_error()


def load_and_process_checkpoints(base_dir):
    """Load and process MLP parameters from all checkpoints"""
    mlp_params = []
    checkpoint_steps = []

    # Get all checkpoint directories
    checkpoints = sorted([d for d in os.listdir(base_dir)
                          if d.startswith('checkpoint-') and os.path.isdir(os.path.join(base_dir, d))],
                         key=lambda x: int(x.split('-')[1]))

    print(f"Found {len(checkpoints)} checkpoint directories")

    for checkpoint in checkpoints:
        step = int(checkpoint.split('-')[1])
        checkpoint_dir = os.path.join(base_dir, checkpoint)

        # Check if safetensors file exists
        if not os.path.exists(os.path.join(checkpoint_dir, 'model.safetensors')):
            print(f"Skipping checkpoint {step}: model.safetensors not found")
            continue

        try:
            print(f"\nProcessing checkpoint {step}...")

            # Load model with safetensors format
            model = GPT2Model.from_pretrained(
                checkpoint_dir,
                local_files_only=True,
                use_safetensors=True  # Explicitly use safetensors format
            )

            # Extract first layer MLP c_fc weights
            mlp_weight = model.h[0].mlp.c_fc.weight.detach().numpy()
            print(f"Extracted weights shape: {mlp_weight.shape}")

            # Flatten weight matrix
            flat_weights = mlp_weight.reshape(-1)
            print(f"Flattened weights shape: {flat_weights.shape}")

            mlp_params.append(flat_weights)
            checkpoint_steps.append(step)

            # Clean GPU memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing checkpoint {step}: {str(e)}")
            continue

    if not mlp_params:
        raise ValueError("No checkpoints were successfully loaded")

    print(f"\nSuccessfully loaded {len(mlp_params)} checkpoints")
    return np.array(mlp_params), checkpoint_steps


def visualize_trajectory(mlp_params, checkpoint_steps, save_path=None):
    """Visualize the MLP parameter training trajectory"""
    # Perform PCA dimensionality reduction
    pca = PCA(n_components=2)
    mlp_params_pca = pca.fit_transform(mlp_params)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Create color mapping
    colors = plt.cm.viridis(np.linspace(0, 1, len(checkpoint_steps)))

    # Plot scatter points
    scatter = plt.scatter(mlp_params_pca[:, 0], mlp_params_pca[:, 1],
                          c=range(len(checkpoint_steps)),
                          cmap='viridis',
                          s=100)

    # Add trajectory arrows
    for i in range(len(checkpoint_steps) - 1):
        arr = FancyArrowPatch(
            (mlp_params_pca[i, 0], mlp_params_pca[i, 1]),
            (mlp_params_pca[i + 1, 0], mlp_params_pca[i + 1, 1]),
            arrowstyle='-|>',
            mutation_scale=15,
            color=colors[i],
            alpha=0.6
        )
        plt.gca().add_patch(arr)

    # Add labels and title
    plt.title('GPT-2 MLP Parameter Training Trajectory', fontsize=14, pad=20)
    plt.xlabel(f'PC1 (Variance Explained: {pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'PC2 (Variance Explained: {pca.explained_variance_ratio_[1]:.2%})', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Training Steps', fontsize=10)

    # Add step annotations (only show 6 evenly spaced steps)
    annotation_indices = np.linspace(0, len(checkpoint_steps) - 1, 6, dtype=int)
    for i in annotation_indices:
        plt.annotate(f'Step {checkpoint_steps[i]}',
                     (mlp_params_pca[i, 0], mlp_params_pca[i, 1]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.7)

    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return mlp_params_pca, pca


def main():
    # Set base directory
    base_dir = "/root/autodl-tmp/model_training/gpt2_training_20250113_0029"

    print(f"Starting analysis of checkpoints in: {base_dir}")

    # Load and process checkpoints
    print("Loading checkpoints...")
    mlp_params, checkpoint_steps = load_and_process_checkpoints(base_dir)

    # Visualize training trajectory
    print("\nGenerating visualization...")
    mlp_params_pca, pca = visualize_trajectory(mlp_params, checkpoint_steps,
                                               save_path="mlp_trajectory.png")

    # Print variance ratios
    print("\nPCA Explained Variance Ratios:")
    print(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")


if __name__ == "__main__":
    main()