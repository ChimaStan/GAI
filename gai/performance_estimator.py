import csv
import random
import torch
import argparse
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageNet
from gai import GAI, Classifier, Attack, Similarity
from utilities import save_result, plot_accuracy_vs_epsilon


class AdversAccuracyEstimator:
    """
    Estimates the adversarial accuracy of the implemented GAI algorithm.

    Attributes:
        gai (GAI): An instance of GAI class for generating adversarial images/examples.
        dataloader (DataLoader): A PyTorch DataLoader object for supplying data batches.
        device (torch.device): The computation device (e.g., 'cuda' or 'cpu').
        save_path (str): Path to save the results of the estimation.
        save_func (callable): Function used to save the estimation results. Defaults to `save_result`.
    """
    def __init__(
            self, 
            gai,
            dataloader,
            device,
            save_path,
            save_func = save_result
    ):

        self.gai = gai
        self.device = device
        self.dataloader = dataloader
        self.save_path = save_path
        self.save_func = save_func

    def estimate(self, epsilon, min_ssim):
        """
        Evaluates the adversarial accuracy for a given epsilon and minimum SSIM constraint.

        Args:
            epsilon (float): The maximum perturbation allowed for adversarial examples.
            min_ssim (float): Minimum Structural Similarity Index (SSIM) to ensure adversarial examples are perceptually similar.

        Returns:
            float: The adversarial accuracy as a ratio of successfully misclassified adversarial examples to total samples.
        """

        total_correct = 0
        total_samples = 0
        
        for images, labels in self.dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            for orig_img, orig_label in zip(images, labels):

                # Add batch dimension (required for processing single images in most models)
                orig_img = orig_img.unsqueeze(0)  # Shape: (1, C, H, W)

                # Generate a random adversarial target different from the original label
                all_classes = torch.arange(0, 1000).to(self.device)
                target_pool = all_classes[all_classes != orig_label]
                advers_target = target_pool[torch.randint(0, len(target_pool), (1,))]

                # Generate adversarial example
                advers_example = self.gai.gen_advers_example(orig_img, advers_target, epsilon, min_ssim)

                # Make predictions on the original image and on the adversarial example
                _, orig_pred_class  = self.gai.predict_class(orig_img)
                _, adv_pred_class   = self.gai.predict_class(advers_example)
                
                if adv_pred_class.item() == advers_target.item():
                    total_correct += 1
                total_samples += 1

                self.save_func(
                    self.save_path,
                    epsilon,
                    min_ssim,             
                    advers_example, 
                    advers_target, 
                    adv_pred_class, 
                    orig_img, 
                    orig_pred_class,
                    orig_label, 
                    idx=total_samples - 1,
                )                

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"Adversarial Accuracy: {accuracy * 100:.2f}%")

        return accuracy
        
                
if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument('-dsp', '--dataset_path', required = True,
                        help = 'Path to dataset e.g., to ImageNet dataset')
    
    parser.add_argument('-dld', '--data_loader', default = 'ImageNet',
                        help = 'Exact name of dataset loader, e.g., ImageNet for the ImageNet dataset')    
    
    parser.add_argument('-spn', '--split_name', default = 'val',
                        help = 'Name of partition e.g., train, val, as applicable to your dataset')
    
    parser.add_argument('-spp', '--split_prop', type = float, default = 0.2,
                        help = 'To use only randomly selected, e.g., 0.2 proportion of samples from the chosen partition  indicated in --spn')

    parser.add_argument('-sed', '--random_seed', type = int, default = 42,
                        help = 'Seed for random selection of samples')    
    
    parser.add_argument('-clf', '--classifier',  default = 'resnet50',
                        help = 'Name of classifier')

    parser.add_argument('-atk', '--attack',  default = 'fgsm',
                        help = 'Name of adversarial attack algorithm.')

    parser.add_argument('-sim', '--sim',  default = 'ssim',
                        help = 'Name similarity measure.')

    parser.add_argument('-msl', '--min_ssim_list', required = True,
                        help = 'Minimum structural similarity index value')

    parser.add_argument('-epl', '--epsilon_list', required = True,
                        help = 'Maximum epsilon')  

    parser.add_argument('-svp', '--save_path', required = True,
                        help = 'Path to save results.')
    

    args = parser.parse_args()    

    # Convert bash-passed comma-separated string (e.g., epsilon_list="0.01,0.007") to a list
    epsilon_list = list(map(float, args.epsilon_list.split(',')))   
    min_ssim_list = list(map(float, args.min_ssim_list.split(',')))


    # Define the classification model
    classifier = Classifier(args.classifier)
    device = classifier.device

    # Instantiate adversarial image generation class
    gai = GAI(
    attack=Attack(args.attack).attack, 
    sim=Similarity(args.sim).metric, 
    model=classifier.model,
    criterion=classifier.criterion
    )

    # Load dataset
    if args.data_loader == 'ImageNet':
        dataset = ImageNet(root=args.dataset_path, split=args.split_name, transform=classifier.transform)
    else:
        raise ValueError(f"Dataset loader {args.loader} not found.")
    dloader = DataLoader(dataset, batch_size=max(8, len(dataset)), shuffle=False)

    # Randomly select indices for a subset of the dataset to use for estimating performance
    if args.split_prop > 0.0:    
        total_indices = list(range(len(dataset)))
        random.seed(args.seed)
        subset_indices = random.sample(total_indices, int(args.split_prop * len(total_indices)))
        subset = Subset(dataset, subset_indices)
        if len(subset) > 0:
            dloader = DataLoader(subset, batch_size=max(8, len(subset)), shuffle=False)
        
    estim =  AdversAccuracyEstimator(gai, dloader, device, args.save_path)

    # Estimate adversarial accuracy and write results
    csv_path = "adversarial_accuracy_results.csv"
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if file.tell() == 0:
            writer.writerow(["min_ssim", "epsilon", "advers_accuracy"])

        for min_ssim in min_ssim_list:
            perf = []
            for epsilon in epsilon_list:
                advers_accuracy = estim.estimate(torch.tensor(epsilon), min_ssim)
                perf.append(advers_accuracy)
                writer.writerow([min_ssim, epsilon, advers_accuracy])
            
            plot_accuracy_vs_epsilon(perf, epsilon_list, min_ssim, args.save_path)