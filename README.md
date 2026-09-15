🌍 DeepGlobe AI | Land Cover Segmentation

An end-to-end semantic segmentation application for satellite imagery using DeepLabV3+ with a ResNet-50 encoder and ImageNet transfer learning.

The application takes a satellite image as input and predicts pixel-level land-cover classes, then presents the segmentation mask, overlay, and class distribution through an interactive Streamlit interface.

🚀 Live Demo

Try the deployed application:
https://deepglobe-land-cover-7rgh2nxjyybusuvpluyne2.streamlit.app/

📌 Project Overview

Satellite imagery contains large amounts of geographic information that can be difficult to analyze manually. This project uses deep learning-based semantic segmentation to automatically classify each pixel of a satellite image into a land-cover category.

The project covers the complete machine-learning workflow:

Dataset preparation

Image and mask preprocessing

Train/validation/test splitting

Transfer learning

Semantic segmentation model training

Model evaluation

Model hosting

Streamlit application development

Cloud deployment

🧠 Model

Component

Configuration

Architecture

DeepLabV3+

Encoder

ResNet-50

Encoder Weights

ImageNet

Input Channels

3 (RGB)

Output Classes

7

Input Resolution

512 × 512

Framework

PyTorch

Segmentation Library

segmentation-models-pytorch

🗺️ Land-Cover Classes

The model predicts seven DeepGlobe land-cover classes:

Urban

Agriculture

Rangeland

Forest

Water

Barren

Unknown

📊 Dataset

The project uses the DeepGlobe Land Cover Classification Dataset.

The labeled data was divided into:

Training: 562 images

Validation: 120 images

Test: 121 images

The images are RGB satellite images paired with pixel-level segmentation masks.

⚙️ Training Configuration

Image size: 512 × 512

Batch size: 4

Epochs: 10

Optimizer: AdamW

Learning rate: 1e-4

Weight decay: 1e-4

Loss: Dice Loss + Cross Entropy

Scheduler: ReduceLROnPlateau

Training hardware: NVIDIA T4 GPU

Loss Function

Total Loss = 0.5 × Dice Loss + 0.5 × Cross Entropy

📈 Model Performance

Best Validation Result

Validation mIoU: 0.4778

Best epoch: Epoch 9

Test Result

Test mIoU: 0.4224

Per-Class Test IoU

Class

IoU

Urban

0.6319

Agriculture

0.7895

Rangeland

0.3133

Forest

0.4862

Water

0.3542

Barren

0.3816

Unknown

0.0000

🖥️ Application Features

📤 Satellite image upload

🧠 DeepLabV3+ semantic segmentation

🗺️ Predicted land-cover segmentation map

🔀 Segmentation overlay on the original image

📊 Class distribution

🎨 Land-cover legend

⚡ CPU-compatible inference

🌐 Public cloud deployment

🔄 Application Workflow

Satellite Image
       ↓
Image Preprocessing
       ↓
Resize to 512 × 512
       ↓
ImageNet Normalization
       ↓
DeepLabV3+ + ResNet-50
       ↓
Pixel-Level Class Prediction
       ↓
Segmentation Mask
       ↓
Overlay + Class Distribution
       ↓
Streamlit Web Interface

☁️ Deployment Architecture

The trained model is hosted separately from the GitHub source code because the .pth file is approximately 107 MB.

GitHub
  │
  │  app.py + requirements.txt
  ↓
Streamlit Community Cloud
  │
  │  downloads model at runtime
  ↓
Hugging Face
  │
  │  best_deeplabv3plus.pth
  ↓
DeepLabV3+ Model
  │
  ↓
Satellite Image Prediction

Model Repository

https://huggingface.co/shreyas-kocheri/deeplabv3plus-deepglobe

🛠️ Tech Stack

Python

PyTorch

Torchvision

Segmentation Models PyTorch

NumPy

Pillow

Streamlit

Hugging Face Hub

Git & GitHub

📂 Repository Structure

DeepGlobe-Land-Cover/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

The trained model is hosted on Hugging Face rather than stored directly in this GitHub repository.

▶️ Run Locally

1. Clone the repository

git clone https://github.com/shreyas-kocheri/DeepGlobe-Land-Cover.git
cd DeepGlobe-Land-Cover

2. Install dependencies

pip install -r requirements.txt

3. Run the Streamlit application

python -m streamlit run app.py

The application will open in your browser.

🔗 Project Links

Live Demo: https://deepglobe-land-cover-7rgh2nxjyybusuvpluyne2.streamlit.app/

GitHub Repository: https://github.com/shreyas-kocheri/DeepGlobe-Land-Cover

Hugging Face Model: https://huggingface.co/shreyas-kocheri/deeplabv3plus-deepglobe

🎯 Key Learning Outcomes

This project demonstrates practical experience with:

Semantic segmentation

Transfer learning

Convolutional neural networks

DeepLabV3+ architecture

ResNet-50

Pixel-level image classification

Image preprocessing and normalization

Model evaluation using IoU and mIoU

PyTorch model deployment

Hugging Face model hosting

Streamlit application development

Git/GitHub version control

Cloud deployment

👨‍💻 Author

Shreyas Kocheri

GitHub: https://github.com/shreyas-kocheri

⭐ If you find this project useful, consider giving the repository a star!
