[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red)](https://keras.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Galaxy10_DECaLS-purple)](https://astronn.readthedocs.io/en/latest/galaxy10.html)
[![Model](https://img.shields.io/badge/Model-CNN-brightgreen)](#)


# <p align="center">MIRAI – Galaxy Morphological Identification and Recognition using Artificial Intelligence</p>

### <p align="center">Summary:</p>
<p align="center">I - Objective</p>
<p align="center">II - Abstract</p>
<p align="center">III - Plan</p>
<p align="center">IV - Methodology</p>
<p align="center">V - Results</p>

---

### 🎯 Objective
Develop a convolutional neural network (CNN) to classify galaxies based on their morphological types using the **Galaxy10 DECaLS** dataset, leveraging deep learning to automate visual classification and support astrophysical research on galaxy evolution.

---

### 📖 Abstract
Galaxy morphology—the shape and structure of galaxies—is key to understanding their formation and evolution. Traditional classification relied on human visual inspection (e.g., Hubble’s Tuning Fork), but modern deep-sky surveys generate millions of galaxy images, making manual classification unfeasible.

With the rise of deep learning, particularly **convolutional neural networks (CNNs)**, automated image-based classification has become a powerful tool. Projects like **Galaxy Zoo** inspired large-scale citizen science efforts, but neural networks now match or exceed human-level accuracy in morphology detection.

The **Galaxy10 DECaLS** dataset is a curated subset of galaxies from the **Dark Energy Camera Legacy Survey (DECaLS)**, labeled into 10 morphological classes. These include smooth galaxies, edge-on disks, spiral galaxies, mergers, and more. Our goal is to train and evaluate a CNN model capable of recognizing these classes with high accuracy, helping astronomers process massive datasets from upcoming surveys like **LSST**.

---

## 🧑‍💻 Setup

### Linux/Mac
```bash
python -m venv venv
source venv/bin/activate
```
### Windows
```bash
python -m venv venv
venv\Scripts\activate  
```

### Requirements
```bash
pip install --upgrade -r requirements.txt
```

### 🔣 Methodology
Scientific Exploration
🔭 Understanding galaxy morphology and how it relates to formation and evolution (e.g., elliptical vs spiral, mergers).
📊 Studying physical parameters (e.g., color index, size, brightness) associated with morphological classes.

#### Dataset
📦 Galaxy10 DECaLS:
Available at Galaxy10 - AstroNN.
Images: RGB 69x69 pixel thumbnails of galaxies.
Classes: 10 morphological types including smooth, spiral, edge-on, merging, barred, and artifact.

#### Preprocessing
🧹 Normalize pixel values between 0 and 1.
📁 Split into training, validation, and test sets.
🔀 Data augmentation: Rotation, zoom, flipping for robustness.

#### Model Development
🧠 CNN Architecture:

#### Input: 69x69x3 RGB image
Convolutional layers + ReLU + MaxPooling
Dropout for regularization
Dense layers + Softmax output (10 classes)
📈 Metrics: Accuracy, confusion matrix, precision/recall per class.

#### Model Training
⚙️ Training the CNN on the Galaxy10 DECaLS dataset.
📉 Monitoring loss and accuracy per epoch.
🧪 Evaluation on test data.

#### Scientific Contribution
🔬 Providing a pretrained classifier for galaxy morphology usable by astronomers.
📁 Contributing to reproducible workflows in astrophysical image classification.
📚 Exploring misclassified cases to gain insight into ambiguous or transitional morphologies.

## 📊 Results

✅ Model Performance

After training our CNN architecture for 25 epochs with early stopping, we achieved the following performance on the test set:

Accuracy: 63,5%
Loss (categorical cross-entropy): 0.73
Validation Accuracy (best epoch): 68.0%
Training Time: ~6 minutes on GPU

🧠 Misclassification Examples

Some edge-on galaxies were misclassified as smooth due to minimal visible disk structure. Similarly, some mergers were confused with spirals, likely due to overlapping arms or distorted features.

🌌 Interpretation

The CNN demonstrates strong ability to learn distinct morphological patterns from compact 69x69 RGB images. Accuracy above 85% is consistent with benchmarks from existing literature. Data augmentation (especially rotations and flips) improved generalization on test data.
