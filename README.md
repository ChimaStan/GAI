# GAI: Generating Adversarial Image
This project intends to generate an adversarial example, given an image and an adversarial target label, using a white-box (targeted) attack approach.
Initial implementation is based on Fast Gradient Sign Method (FGSM) of attack, using a pretrained ResNet50 classifier and Structural Similarity Index (SSIM) metric as the constraint on the amount of pertubation applied to the input image. Hyperparameter search for the suitable FGSM's epsilon parameter and minimum SSIM value required to maintain perceptual consistency between the pertubed image and input image whilst realising adversarial effect is conducted and evaluated using the ImageNet ILSVRC2012_img_val dataset. 


## Recommended way to run
- Start terminal or PowerShell

- git clone GAI remote repo 

    git clone https://github.com/ChimaStan/GAI.git

- Navigate to the cloned parent directory (GAI)

- Create a Docker image with dependencies installed

    docker build -t gai:env

- Start docker in interactive mode binding GAI and directory of image file(s) you wish to test to the container

    docker run -it -v ./:/app -v /path/to/local/test/images:/imgs gai:env

- Command to generate an adversarial image for a given input

    python gai/demo.py -ifp path_to_image_file -tgt target_label

    your_image_file: path to the image you wish to test (e.g., /imgs/image1.jpg).  
    target_label: Preferrably an integer, in the range [0,999] corresponding to ImageNet ILSVRC dataset's classes.

- To find the generated adversarial image

    Navigate to GAI/data/output/gai. Filenames follow the structure: [your_image_file_without_extension]_[target_label].[extension].

