import csv
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

def load_image_from_file(image_file, transform=None):
    image = Image.open(image_file).convert('RGB')
    if transform:
        image = transform(image).unsqueeze(0)
    return image


def tensor_to_pil(pt_image):
    """
    Converts an image in PyTorch tensor format to a PIL Image.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W) with pixel values in [0, 1].

    Returns:
        PIL.Image.Image: The corresponding PIL image.
    """
    np_image = pt_image.squeeze().permute(1, 2, 0).cpu().numpy()
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return pil_image


def save_result(
    save_path,
    epsilon,
    min_ssim,
    advers_example,
    advers_target,
    adv_pred_class,
    orig_img,
    orig_pred_class,
    orig_label=None,
    idx=0,
):
    """
    Save results including original and adversarial images, and metadata as a CSV file.

    Args:
        save_path (str): Base path to save results.
        epsilon (float): Maximum epsilon value used.
        min_ssim (float): Minimum SSIM value used.
        advers_example (numpy.ndarray): Adversarial image.
        advers_target (int): Target class for the adversarial image.
        adv_pred_class (int): Predicted class for the adversarial image.
        orig_img (numpy.ndarray): Original image.
        orig_pred_class (int): Predicted class of the original image.
        orig_label (int): Label of the original image.
        idx (int): Identifier for the sample.
    """
    save_path = Path(save_path)
    images_path = save_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    orig_img_path = images_path / f"orig-{idx}.png"
    adv_img_path = images_path / f"adv-{idx}.png"
    
    # Convert to PIL-formatted image
    tensor_to_pil(orig_img).save(orig_img_path)
    tensor_to_pil(advers_example).save(adv_img_path)

    # Save metadata to a CSV file
    csv_path = save_path / f"results-ssim{min_ssim}-eps{epsilon}.csv"
    write_header = not csv_path.exists()  # Check if the CSV file needs a header
    with csv_path.open(mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["id", "orig_label", "orig_pred_class", "advers_target", "adv_pred_class"])
        writer.writerow([idx, orig_label, orig_pred_class, advers_target, adv_pred_class])


def plot_accuracy_vs_epsilon(perf, epsilon_list, min_ssim, save_path):
    """
    Plots the adversarial accuracy vs epsilon for a given min_ssim value and saves the plot.

    Args:
        perf (list): List of adversarial accuracy values corresponding to each epsilon.
        epsilon_list (list): List of epsilon values.
        min_ssim (float): The min_ssim value corresponding to the provided accuracy values.
        save_path (str): The path where the plot will be saved.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    plt.plot(epsilon_list, perf, marker='o', label=f'min_ssim={min_ssim}')

    plt.xlabel('Epsilon')
    plt.ylabel('Adversarial Accuracy')
    plt.title(f'Adversarial Accuracy vs Epsilon (min_ssim={min_ssim})')

    plt.legend(title='min_ssim')
    plt.grid(True)

    plot_path = save_path / f"adversarial_accuracy_vs_epsilon-for-minssim-{min_ssim}.png"
    plt.savefig(plot_path)

    # Optional
    # plt.show()

    plt.close()


def display_example(original_image, 
                    adversarial_image, 
                    predicted_class_original, 
                    predicted_class_adversarial, 
                    save_path=None):
    """
    Displays a figure with original and adversarial images along with predicted classes.
    Optionally saves the figure to a file.

    Args:
        original_image: The original image to display (as a numpy array or similar format).
        adversarial_image: The adversarial image to display (as a numpy array or similar format).
        predicted_class_original: The predicted class for the original image.
        predicted_class_adversarial: The predicted class for the adversarial image.
        save_path: Path to save the figure. If None, the figure is not saved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].axis('off')
    axes[0].set_title(r'$\bf{Input\ image}$', fontsize=14)
    axes[0].text(0.5, -0.2, 
                 f'Predicted class: {predicted_class_original}', 
                 fontsize=12, 
                 ha='center', va='center', transform=axes[0].transAxes)

    # Adversarial image
    axes[1].imshow(adversarial_image)
    axes[1].axis('off')
    axes[1].set_title(r'$\bf{Adversarial\ image}$', fontsize=14)
    axes[1].text(0.5, -0.2, 
                 f'Predicted class: {predicted_class_adversarial}', 
                 fontsize=12, 
                 ha='center', va='center', transform=axes[1].transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
