"""
Main script to generate adversarial examples for classification models.
Loads an image, applies an adversarial attack, and compares the original 
and adversarial predictions.
"""
import torch
import argparse
from pathlib import Path
from gai import GAI, Classifier, Attack, Similarity
from utilities import (load_image_from_file, tensor_to_pil, 
                       display_example, get_mean_std_from_transform)


if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument('-ifp', '--image_file_path', required = True,
                        help = 'Path to an image file.')
    
    parser.add_argument('-tgt', '--advers_target', type = int, required = True,
                        help = 'Adversarial target label. Integer of range [0,999] or string available in imagenet classes.')
        
    parser.add_argument('-ild', '--image_loader', default = 'load_image_from_file',
                        help = 'Exact name of function to read and load image file.')    
    
    parser.add_argument('-clf', '--classifier',  default = 'resnet50',
                        help = 'Name of classifier')

    parser.add_argument('-atk', '--attack',  default = 'fgsm',
                        help = 'Name of adversarial attack algorithm.')

    parser.add_argument('-sim', '--sim',  default = 'ssim',
                        help = 'Name similarity measure.')

    parser.add_argument('-mss', '--min_ssim', type = float, default = 0.95,
                        help = 'Minimum structural similarity index value')
     
    parser.add_argument('-eps', '--epsilon', type = float, default = 0.007,
                        help = 'Maximum epsilon')       

    parser.add_argument('-svp', '--save_path', default = None,
                        help = 'Path to save results.')


    args = parser.parse_args() 

    classifier = Classifier(args.classifier)
    device = classifier.device
    transform = classifier.transform

    # Instantiate adversarial image generation class

    gai = GAI(
        attack=Attack(args.attack).attack, 
        sim=Similarity(args.sim).metric, 
        model=classifier.model,
        criterion=classifier.criterion
    )

    # Load ImageNet class labels
    imagenet_classes_path = Path("./assets/imagenet_classes.txt")

    if not imagenet_classes_path.exists():
        raise FileNotFoundError(f"{imagenet_classes_path} does not exist.")
    with open(imagenet_classes_path, "r") as f:
        class_labels = [line.strip() for line in f.readlines()]

    # Get numeric value for the adversarial target class whether provided as an integer or as a string 
    try:
        advers_target_idx = int(args.advers_target)
        if args.advers_target > 999:
            raise ValueError(f"Target class {args.advers_target} is out of bound: must be in range [0,999].")
    except ValueError:
        try:
            # Get the numeric index of the class name
            advers_target_idx = class_labels.index(args.advers_target)
        except ValueError:
            raise ValueError(f"Class '{args.advers_target}' not found in imagenet class_labels.")

    # Load input image 
    valid_loaders = ['load_image_from_file']  # Extend this list if there are more loaders
    if args.image_loader not in valid_loaders:
        raise ValueError(f"Invalid image loader: {args.image_loader}. Expected one of {valid_loaders}.")
    if args.image_loader == 'load_image_from_file':
        orig_img = load_image_from_file(args.image_file_path, transform)

    # Generate adversarial example
    advers_example = gai.gen_advers_example(orig_img, torch.tensor([advers_target_idx]), torch.tensor(args.epsilon), args.min_ssim)

    # Make predictions on the original image and on the adversarial example
    _, orig_pred_class_idx  = gai.predict_class(orig_img)
    _, adv_pred_class_idx   = gai.predict_class(advers_example)

    if args.save_path is not None:
        save_path = Path(args.save_path)
    else:
        fpath = Path(args.image_file_path)
        save_path = Path("./data/output/gai")
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f"{fpath.stem}_{str(args.advers_target)}{fpath.suffix}"
        
    mean, std = get_mean_std_from_transform(transform)

    display_example(
        original_image = tensor_to_pil(orig_img, mean, std),
        adversarial_image = tensor_to_pil(advers_example), 
        predicted_class_original = class_labels[orig_pred_class_idx.item()],
        advers_target_class=class_labels[advers_target_idx], 
        predicted_class_advers = class_labels[adv_pred_class_idx.item()], 
        save_path=save_path
    )
