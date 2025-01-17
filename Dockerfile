# Lightweight Pytorch base image with support for CUDA and cuDNN
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . . 

# Installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run the package
# CMD ["python", "demo.py"]
CMD ["/bin/bash"]

