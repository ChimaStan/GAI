import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as compare_ssim

class GAI:
    """
    General Adversarial Interface (GAI) for generating and evaluating adversarial examples.

    Attributes:
        attack (function): Adversarial attack method.
        sim (function): Similarity metric function.
        model (torch.nn.Module): Neural network model to attack.
        criterion (torch.nn.Module): Loss function for gradient computation.
    """
    def __init__(self, attack, sim, model, criterion):
        """
        Initializes the GAI class.

        Args:
            attack (function): Function to perform adversarial attacks.
            sim (function): Function to compute similarity between images.
            model (torch.nn.Module): Model to be evaluated.
            criterion (torch.nn.Module): Loss function for adversarial training.
        """        
        self.attack = attack
        self.sim = sim
        self.model = model
        self.criterion = criterion
        self.model.eval()

    def compute_gradient(self, image, target):
        """
        Computes the gradient of the loss with respect to the input image.

        Args:
            image (torch.Tensor): Input image.
            target (torch.Tensor): Target class label.

        Returns:
            torch.Tensor: Gradient of the loss with respect to the image.
        """
        image.requires_grad = True
        output = self.model(image)
        loss = self.criterion(output, target)
        self.model.zero_grad()
        loss.backward()
        gradient = image.grad.data
        return gradient        

    def compute_advers_img(self, image, target_class, epsilon):
        """
        Generates an adversarial image by applying a perturbation.

        Args:
            image (torch.Tensor): Original image.
            target_class (torch.Tensor): Target class label.
            epsilon (float): Perturbation magnitude.

        Returns:
            tuple: Adversarial image and gradient.
        """
        gradient = self.compute_gradient(image, target_class)
        advers_img = self.attack(image, epsilon, gradient)
        return advers_img, gradient

    def ensure_similarity(self, image, advers_img, gradient, epsilon, min_ssim, factor=2):
        """
        Ensures the adversarial image satisfies a similarity threshold.

        Args:
            image (torch.Tensor): Original image.
            advers_img (torch.Tensor): Adversarial image.
            gradient (torch.Tensor): Gradient used for perturbation.
            epsilon (float): Perturbation magnitude.
            min_ssim (float): Minimum similarity threshold.
            factor (int): Reduction factor for epsilon.

        Returns:
            torch.Tensor: Adjusted adversarial image.
        """
        similarity = self.sim(image, advers_img)
        if similarity < min_ssim:
            print(f"SSIM ({similarity:.2f}) below similarity threshold of {min_ssim}. Reducing epsilon by {factor}.")
            epsilon /= factor
            advers_img = self.attack(image, epsilon, gradient)
            self.ensure_similarity(image, advers_img, gradient, epsilon, min_ssim)
        return advers_img
    
    def gen_advers_example(self, image, target_class, epsilon, min_ssim):
        """
        Generates an adversarial example satisfying similarity constraints.

        Args:
            image (torch.Tensor): Original image.
            target_class (torch.Tensor): Target class label.
            epsilon (float): Perturbation magnitude.
            min_ssim (float): Minimum similarity threshold.

        Returns:
            torch.Tensor: Generated adversarial example.
        """
        advers_img, gradient = self.compute_advers_img(image, target_class, epsilon)
        advers_img = self.ensure_similarity(image, advers_img, gradient, epsilon, min_ssim)    
        return advers_img
    
    def predict_class(self, image):
        """
        Predicts the class of the given image.

        Args:
            image (torch.Tensor): Input image.

        Returns:
            tuple: Logits and predicted class.
        """        
        with torch.no_grad():
            output = self.model(image)
        return output.max(1)    
       
       
class Classifier:
    """
    Encapsulates pre-trained classifiers, such as ResNet-50, for adversarial evaluation.
    """    
    SUPPORTED_MODELS = ['resnet50']  # Extend this list as you add more models

    def __init__(self, model_name, device=None):
        """
        Initializes the Classifier class.

        Args:
            model_name (str): Name of the model (e.g., 'resnet50').
            device (str, optional): Device for computation (e.g., 'cuda:0').
        """
        if model_name.lower() not in self.SUPPORTED_MODELS:
            raise ValueError(f"Invalid model name: {model_name}. Supported models are: {self.SUPPORTED_MODELS}")

        # Set device (use GPU if available, otherwise CPU)
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model_name = model_name.lower()
        self.model = model_name.lower()  # call the model.setter

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        # Dynamically call the corresponding model-loading method
        model_method = f'_load_{value}'
        if hasattr(self, model_method):
            self._model = getattr(self, model_method)()
        else:
            raise ValueError(f"Model {value} is not implemented.")

    def _load_resnet50(self):
        """
        Load a pre-trained ResNet-50 model and set up appropriate image transformations.
        """        
        model = models.resnet50(pretrained=True)
        model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.criterion = nn.CrossEntropyLoss()
        
        return model.to(self.device)
    
    # Extend or integrate with other solutions.
    
class Attack:
    """
    Encapsulates adversarial attack methods, such as the Fast Gradient Sign Method (FGSM).
    """
    SUPPORTED_ATTACKS = ['fgsm']  # Add new attack methods here in the future
    
    def __init__(self, attack_name):
        # Ensure the attack name is valid
        if attack_name.lower() not in self.SUPPORTED_ATTACKS:
            raise ValueError(f"Invalid attack name: {attack_name}. Supported attacks are: {self.SUPPORTED_ATTACKS}")
        
        # Set the attack method dynamically
        self.attack_name = attack_name.lower()
        self.attack = attack_name.lower()  # This will call the setter

    @property
    def attack(self):
        return self._attack

    @attack.setter
    def attack(self, value):
        # Dynamically call the corresponding attack method
        attack_method = f'_{value}'
        if hasattr(self, attack_method):
            self._attack = getattr(self, attack_method)
        else:
            raise ValueError(f"Attack method {value} is not implemented.")
    
    def _fgsm(self, image, epsilon, gradient):
        perturbation = epsilon * gradient.sign()
        advers_img = torch.clamp(image + perturbation, 0, 1)
        return advers_img.detach()

    # Extend or integrate with other solutions.

class Similarity:
    """
    Encapsulates similarity metrics for evaluating adversarial example quality, 
    such as structural similarity index (SSIM), which tends to correlate with human perception.
    """
    SUPPORTED_METRICS = ['ssim']  # Add new similarity metrics here in the future
    
    def __init__(self, metric_name):
        # Ensure the metric name is valid
        if metric_name.lower() not in self.SUPPORTED_METRICS:
            raise ValueError(f"Invalid metric name: {metric_name}. Supported metrics are: {self.SUPPORTED_METRICS}")
        
        # Set the metric method dynamically
        self.metric_name = metric_name.lower()
        self.metric = metric_name.lower()  # This will call the setter

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        # Dynamically call the corresponding metric method
        metric_method = f'_{value}'
        if hasattr(self, metric_method):
            self._metric = getattr(self, metric_method)
        else:
            raise ValueError(f"Similarity metric {value} is not implemented.")
    
    def _ssim(self, original, adversarial):
        original_np = original.squeeze().permute(1, 2, 0).cpu().numpy()
        adversarial_np = adversarial.squeeze().permute(1, 2, 0).cpu().numpy()
        ssim_value = compare_ssim(original_np, adversarial_np, multichannel=True)
        return ssim_value

    # Extend or integrate with other solutions.
