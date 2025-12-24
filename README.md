
---

# 🌾 Paddy Disease Classification using PyTorch and ResNet-18

This repository contains a **deep learning project** focused on classifying diseases in **paddy (rice) leaves** using **transfer learning** with a **pre-trained ResNet-18 model**.
Built with **PyTorch**, the model achieves **high accuracy** by leveraging powerful image features learned from large-scale datasets (ImageNet).

---

## 📋 Table of Contents

* [Project Overview](#project-overview)
* [Workflow Pipeline](#workflow-pipeline)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Getting Started](#getting-started)
* [Installation](#installation)
* [Usage](#usage)
* [Results](#results)
* [Code Structure](#code-structure)

---

## 📖 Project Overview

The objective of this project is to develop a **reliable image classification system** capable of detecting various **paddy leaf diseases** automatically.
Such systems can play a major role in **precision agriculture**, helping farmers identify crop diseases early and improve yield outcomes.

This notebook implements an end-to-end workflow for deep learning–based disease classification, covering:

* **Dataset Acquisition** from Kaggle
* **Image Preprocessing & Augmentation** for robust model training
* **Model Training & Fine-tuning** using ResNet-18
* **Performance Evaluation** on validation data

The final trained model demonstrates strong generalization performance and highlights the power of **transfer learning** for agricultural image classification tasks.

---

## 📊 Workflow Pipeline

```
+----------------------------+
|    1. Dataset Download     |
|    (Kaggle API)            |
+----------------------------+
             |
             ▼
+----------------------------+
|  2. Image Preprocessing    |
|  - Resize (128x128)        |
|  - Random Rotations/Flips  |
|  - Normalization           |
+----------------------------+
             |
             ▼
+----------------------------+
|   3. Data Split & Loading  |
|  - 80% Train / 20% Val     |
|  - PyTorch DataLoaders     |
+----------------------------+
             |
             ▼
+----------------------------+
|   4. Model Setup           |
|  - Pre-trained ResNet-18   |
|  - Custom Classifier Head  |
|  - Dropout Regularization  |
+----------------------------+
             |
             ▼
+----------------------------+
|   5. Model Training        |
|  - Adam Optimizer          |
|  - CrossEntropy Loss       |
|  - 50 Epochs (GPU/CPU)     |
+----------------------------+
             |
             ▼
+----------------------------+
|   6. Model Evaluation      |
|  - Validation Accuracy     |
|  - Loss Monitoring         |
+----------------------------+
             |
             ▼
+----------------------------+
|   7. Final Results         |
|  - Accuracy: 96.7%         |
|  - Loss: 0.19              |
+----------------------------+
```

---

## 🌱 Dataset

The dataset used is from the **Kaggle Paddy Disease Classification Challenge**.

**Details:**

* **Source:** Kaggle (Paddy Disease Classification dataset)
* **Classes:** 10 total (9 diseased + 1 healthy)
* **Image Size:** Resized to 128×128 pixels
* **Split:** 80% for training, 20% for validation
* **Format:** Folder structure compatible with `torchvision.datasets.ImageFolder`

This dataset provides a diverse and balanced collection of paddy leaf images for effective supervised learning.

---

## 🧠 Model Architecture

This project utilizes **Transfer Learning** with **ResNet-18**, a convolutional neural network pre-trained on ImageNet.

**Architecture Highlights:**

* **Base Model:** ResNet-18 (`torchvision.models.resnet18`)
* **Feature Extractor:** Frozen convolutional layers from ImageNet
* **Classifier Head:** Custom `nn.Sequential` block containing:

  * `Dropout(p=0.1)` for regularization
  * `Linear(in_features=512, out_features=10)` for 10 disease categories

This design ensures efficient training while maintaining excellent generalization on unseen data.

---

## 🚀 Getting Started

### 🔧 Prerequisites

* Python 3.8+
* Kaggle Account + API key (`kaggle.json`)
* Jupyter Notebook or JupyterLab

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<Sammarth14052018>/Paddy-Disease-Classifier.git
cd Paddy-Disease-Classifier

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt should include:**

```
torch  
torchvision  
opendatasets  
numpy  
matplotlib  
tqdm  
opencv-python  
pickle-mixin  
```

**Set up Kaggle API:**

1. Download your `kaggle.json` from your Kaggle account settings.
2. Place it in the root directory.
3. The first notebook run will prompt for your Kaggle credentials.

---

## 💻 Usage

1. Launch Jupyter Notebook or JupyterLab.
2. Open the notebook file (`Paddy_Disease_Classifier.ipynb`).
3. Run all cells sequentially — the notebook will:

   * Download the dataset via Kaggle API
   * Preprocess and augment images
   * Initialize and train the ResNet-18 model
   * Evaluate performance and display accuracy/loss metrics

During training, you’ll see epoch-by-epoch progress with live validation metrics.

---

## 📈 Results

**Training Summary:**

* **Epochs:** 50
* **Optimizer:** Adam
* **Loss Function:** CrossEntropyLoss
* **Validation Accuracy:** **96.7%**
* **Validation Loss:** **0.1895**

This demonstrates that the model generalizes effectively and can accurately detect diseases across multiple categories.

Sample training batch (augmented images):

*(Insert example plot if available)*

---

## 🔍 Code Structure

| Section              | Description                                                                |
| -------------------- | -------------------------------------------------------------------------- |
| **Data Preparation** | Loads data, applies augmentations, and creates PyTorch DataLoaders         |
| **Model Definition** | Initializes ResNet-18, replaces classifier head, and sets device (GPU/CPU) |
| **Training Setup**   | Configures loss, optimizer, and learning parameters                        |
| **Training Loop**    | Trains model for 50 epochs and logs metrics                                |
| **Evaluation**       | Calculates final validation accuracy and loss                              |

---

## 🏗️ Future Improvements

* Implement **ResNet-50** or **EfficientNet** for deeper feature extraction
* Add **Grad-CAM** visualizations to interpret model predictions
* Deploy as a **Streamlit web app** for real-time disease detection
