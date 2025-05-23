# 🎨 Neural Style Transfer with VGG-19

This project implements **Neural Style Transfer (NST)** using a pre-trained **VGG-19** Convolutional Neural Network. The goal is to generate a new image that combines the structural **content** of one image with the artistic **style** of another — creating visually compelling, stylized artwork.

---

## 🧠 What is Neural Style Transfer?

Neural Style Transfer is a deep learning technique that blends the **content of a source image** with the **artistic style of a reference image**.

This is achieved by:
- Extracting high-level **content features** and low-level **style features** using a CNN.
- Optimizing a new image to minimize both content and style loss.

> Originally proposed by Gatys et al. (2015), NST has enabled creative applications from digital art to video stylization.

---

## ⚙️ How It Works

### **Input**
- 🖼️ Content image (structure)
- 🎨 Style image (artistic influence)

### **Feature Extraction**
- Use a **pre-trained VGG-19** to extract hierarchical features.

### **Representations**
- **Content**: Higher-layer activations.
- **Style**: Gram matrices of lower-layer activations.

### **Loss Functions**
- **Content Loss**: Difference between content features.
- **Style Loss**: Difference between Gram matrices.

### **Optimization**
- Initialize with the content image or white noise.
- Use **gradient descent** (L-BFGS or Adam) to minimize total loss.

### **Output**
- A stylized image that preserves content structure and adopts artistic style.

---

## 🔧 Hyperparameters

| Parameter              | Value     |
|------------------------|-----------|
| Content weight (α)     | 1         |
| Style weight (β)       | 1e9       |
| Learning rate          | 0.003     |
| Iterations             | 2000      |
| Activation function    | ReLU      |

**Style Layer Weights:**
- `conv1_1`: 1.0  
- `conv2_1`: 0.75  
- `conv3_1`: 0.2  
- `conv4_1`: 0.2  
- `conv5_1`: 0.2  

---

## 🖼️ Results

- The project demonstrates successful style transfer on diverse image pairs.
- **Multi-style transfer** is also supported.
- See results in `data/sample_images/`.

| Content | Style | Output |
|--------|-------|--------|
| ![](data/sample_images/content.jpg) | ![](data/sample_images/style.jpg) | ![](data/sample_images/output.jpg) |

---


## 🙌 Credits

- [Leon Gatys et al. (2015)](https://arxiv.org/abs/1508.06576) — *A Neural Algorithm of Artistic Style*
- Kashyap, K., Fargose, S., Garg, M., & Nair, S. (2025) — *Dynamic Neural Style Transfer for Artistic Image Generation using VGG19*
- PyTorch, VGG-19 pretrained model

---
