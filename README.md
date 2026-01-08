# **Medical Image Segmentation and Classification for COVID-19 Diagnosis**

This repository contains the implementation of an automated COVID-19 diagnosis system using chest X-ray images, developed as part of a Medical Image Analysis project. The project combines deep learning approaches for both image classification and segmentation to identify COVID-19, Healthy, and Non-COVID (Viral Pneumonia) cases, while also providing visual segmentation of infected lung regions.

## **Project Overview**

This project aims to develop an automated diagnostic system for analyzing chest X-ray images to detect COVID-19 infections. The system addresses the challenge of rapid and accurate COVID-19 screening by leveraging state-of-the-art deep learning architectures for both classification (disease diagnosis) and segmentation (infection region identification). The project compares multiple deep learning models, including traditional CNNs (ResNet, VGG), attention-based architectures (Attention U-Net, R2-Attention U-Net), and vision-language models (CLIP, CLIPSeg), evaluating their performance using comprehensive metrics.

## **Dataset**

The dataset used is the **COVID-19 Radiography Database** from Kaggle (`tawsifurrahman/covid19-radiography-database`), consisting of chest X-ray images with corresponding lung infection masks. Key characteristics:

* **Classes:** Three categories - COVID-19, Healthy (Normal), and Non-COVID (Viral Pneumonia)
* **Data Structure:** Each class contains:
  - `images/`: Chest X-ray images in PNG format
  - `masks/`: Binary segmentation masks highlighting infected regions
* **Image Format:** Grayscale chest X-ray images standardized to 256×256 pixels
* **Data Splits:** Train/Validation/Test splits are pre-configured using CSV files in `dataset/splits/`

**Download the dataset:** Run the installation script to automatically download and prepare the dataset from Kaggle:

```bash
python utils/install_dataset.py
```

Alternatively, manually download from [Kaggle COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and place it in the `dataset/` directory.

## **Methods and Models**

### **Pre-processing**
* **Image Normalization:** Images are normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
* **Data Augmentation:** Albumentations library is used for training augmentation including random rotations, horizontal flips, brightness/contrast adjustments
* **Resizing:** All images are resized to 256×256 for consistency across models
* **Mask Processing:** Binary masks are normalized to [0, 1] range for segmentation training

### **Classification Models**

**ResNet (ResNet18 & ResNet50):**
  * Deep residual networks with skip connections to prevent vanishing gradient problems
  * Transfer learning: Pre-trained on ImageNet, fine-tuned using two-stage approach
  * Stage 1: Feature extraction (frozen backbone, train classification head only)
  * Stage 2: Full fine-tuning with very low learning rate
  * Strengths: Efficient training, good generalization with residual connections
  * Architecture: 18-layer (ResNet18) and 50-layer (ResNet50) configurations

**VGG (VGG16 & VGG19):**
  * Classical CNN architecture with small 3×3 convolution filters and deep networks
  * Batch normalization variants used for improved training stability
  * Pre-trained on ImageNet with custom classification head (Dropout + Linear layer)
  * Strengths: Simple architecture, strong feature extraction capabilities
  * Architecture: 16-layer (VGG16) and 19-layer (VGG19) configurations with BatchNorm

**CLIP (Contrastive Language-Image Pre-training):**
  * Vision-language model trained on image-text pairs
  * Zero-shot learning capability using text prompts (e.g., "a chest X-ray showing COVID-19")
  * Fine-tuned for multi-class classification with frozen text encoder
  * Strengths: Leverages semantic understanding, excellent generalization
  * Superior performance due to multi-modal pre-training on diverse medical imagery

### **Segmentation Models**

**ResNetUnet:**
  * Encoder-decoder architecture using ResNet34 as encoder with pre-trained ImageNet weights
  * Skip connections between encoder and decoder for preserving spatial information
  * Strengths: Efficient feature extraction, good balance between accuracy and speed

**Attention U-Net:**
  * Enhanced U-Net with attention gates that focus on relevant regions
  * Attention mechanism helps suppress irrelevant background features
  * Strengths: Improved boundary delineation, better handling of varied infection patterns

**R2U-Net (Recurrent Residual U-Net):**
  * Combines recurrent and residual connections in U-Net architecture
  * Recurrent blocks enable better feature representation through time-unrolling
  * Strengths: Enhanced feature extraction for complex patterns

**R2AttU-Net (Recurrent Residual Attention U-Net):**
  * Integrates attention gates into R2U-Net architecture
  * Combines benefits of recurrence, residual learning, and attention mechanisms
  * Challenges: High computational complexity, potential overfitting on smaller datasets

**CLIPSeg:**
  * Vision-language segmentation model based on CLIP architecture
  * Text-conditioned segmentation using prompts (e.g., "infected lung regions in chest X-ray")
  * Strengths: Semantic segmentation with text guidance, zero-shot capabilities
  * Fine-tuned with task-specific prompts for COVID-19 infection segmentation

## **Results**

### **Classification Performance**

| Model    | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
|----------|--------------|---------------|------------|--------------|
| ResNet18 | 96.83        | 96.84         | 96.83      | 96.82        |
| ResNet50 | 97.36        | 97.36         | 97.36      | 97.36        |
| VGG16    | 98.35        | 98.36         | 98.35      | 98.34        |
| VGG19    | 97.56        | 97.57         | 97.56      | 97.55        |
| **CLIP** | **99.08**    | **99.08**     | **99.08**  | **99.07**    |

### **Segmentation Performance**

| Model          | IoU (%)  | Dice (%) | Pixel Acc (%) | Precision (%) | Recall (%) | F1 Score (%) |
|----------------|----------|----------|---------------|---------------|------------|--------------|
| **ResNetUnet** | **96.58**| **98.23**| **99.17**     | **97.86**     | **98.65**  | **98.23**    |
| AttentionUNet  | 95.94    | 97.86    | 99.01         | 97.97         | 97.86      | 97.86        |
| R2Unet         | 92.25    | 95.81    | 98.08         | 96.61         | 95.33      | 95.81        |
| R2AttUnet      | 79.35    | 87.09    | 94.60         | 91.09         | 86.71      | 87.09        |
| CLIPSeg        | 94.19    | 96.95    | 98.57         | 97.09         | 96.90      | 96.95        |

### **Key Findings**

- **CLIP** achieved the highest classification accuracy (99.08%), demonstrating the power of vision-language pre-training for medical image analysis
- **ResNetUnet** outperformed all segmentation models with 96.58% IoU and 98.23% Dice score, showing excellent infection region delineation
- **VGG16** showed strong classification performance (98.35%), balancing accuracy and model simplicity
- **AttentionUNet** achieved competitive segmentation results (95.94% IoU), effectively focusing on infected regions through attention mechanisms
- **R2AttUnet** struggled with overfitting due to high complexity, resulting in lower performance (79.35% IoU)
- **CLIPSeg** demonstrated promising text-guided segmentation capabilities (94.19% IoU) despite being a general-purpose model

## **Conclusion**

The project successfully developed and compared multiple deep learning architectures for COVID-19 diagnosis from chest X-ray images. CLIP demonstrated superior classification performance by leveraging vision-language pre-training, while ResNetUnet achieved the best segmentation results through efficient encoder-decoder architecture with skip connections. Future work could focus on:

- Expanding the dataset to include more diverse patient demographics and imaging conditions
- Implementing ensemble methods combining multiple classification and segmentation models
- Developing lightweight models for deployment on mobile and edge devices
- Exploring explainability techniques (Grad-CAM, attention visualization) for clinical interpretability
- Integrating multi-modal data (patient history, lab results) for comprehensive diagnosis

## **Installation**

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/bababyVN/medical-image-segmentation-and-classification.git
```

Navigate to the project directory:

```bash
cd medical-image-segmentation-and-classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## **Download the Dataset**

Install the COVID-19 Radiography Database from Kaggle:

```bash
python utils/install_dataset.py
```

This will automatically download and prepare the dataset in the `dataset/` directory with proper class organization and train/validation/test splits.

## **Download Pre-trained Model Weights**

Download the pre-trained model checkpoints from this [Google Drive link](https://drive.google.com/drive/folders/1tLje0zwL8PTz7-p0IsRwtTH0tPAytbk7).

Unzip the downloaded file and place the weights in the project's working directory (same level as `demo/`, `models/`, etc.).


## **Demo Application**

Launch the interactive Gradio web application for COVID-19 diagnosis:

```bash
python demo/app.py
```

The application provides:
- Interactive chest X-ray image upload
- Model selection for classification and segmentation
- Real-time COVID-19 diagnosis with confidence scores
- Visual segmentation overlay for infected regions
- Adjustable opacity for infection mask visualization

## **Project Structure**

```
medical-image-segmentation-and-classification/
├── dataset/                          # Dataset directory (created after installation)
│   ├── COVID/
│   │   ├── images/
│   │   └── masks/
│   ├── Healthy/
│   │   ├── images/
│   │   └── masks/
│   ├── Non-COVID/
│   │   ├── images/
│   │   └── masks/
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── models/
│   ├── classification_models/       # Classification model architectures
│   │   ├── ResNet.py
│   │   ├── VGG.py
│   │   └── CLIP.py
│   └── segmentation_models/         # Segmentation model architectures
│       ├── ResnetUnet.py
│       ├── AttentionUNet.py
│       ├── R2U_Net.py
│       ├── R2AttU_Net.py
│       └── CLIPSeg.py
├── utils/                           # Utility scripts
│   ├── dataset.py                  # Dataset loaders
│   ├── trainer.py                  # Training scripts
│   ├── tester.py                   # Evaluation scripts
│   ├── helpers.py                  # Helper functions
│   ├── pipeline.py                 # Inference pipeline
│   ├── clip_finetuner.py          # CLIP fine-tuning
│   ├── clip_seg_finetuner.py      # CLIPSeg fine-tuning
│   ├── install_dataset.py         # Dataset download script
│   └── split_dataset.py           # Data splitting utility
├── weights/
├── demo/                            # Gradio web application
│   ├── app.py                      # Main application
│   └── design.css                  # Custom styling
├── notebooks/
│   └── EDA.ipynb                   # Exploratory Data Analysis
├── results/                         # Evaluation results
│   ├── classification_test_results.csv
│   └── segmentation_test_results.csv
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # This file
```

## **References**

[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 770-778, 2016. URL https://arxiv.org/abs/1512.03385.

[2] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In *International Conference on Learning Representations (ICLR)*, 2015. URL https://arxiv.org/abs/1409.1556.

[3] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever. Learning transferable visual models from natural language supervision. In *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 2021. URL https://arxiv.org/abs/2103.00020.

[4] O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, pages 234-241, 2015. URL https://arxiv.org/abs/1505.04597.

[5] O. Oktay, J. Schlemper, L. L. Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz, B. Glocker, and D. Rueckert. Attention U-Net: Learning where to look for the pancreas. *arXiv preprint arXiv:1804.03999*, 2018. URL https://arxiv.org/abs/1804.03999.

[6] M. Z. Alom, M. Hasan, C. Yakopcic, T. M. Taha, and V. K. Asari. Recurrent residual convolutional neural network based on U-Net (R2U-Net) for medical image segmentation. *arXiv preprint arXiv:1802.06955*, 2018. URL https://arxiv.org/abs/1802.06955.

[7] T. Lüddecke and A. Ecker. Image segmentation using text and image prompts. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 7086-7096, 2022. URL https://arxiv.org/abs/2112.10003.

[8] M. E. H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M. A. Kadir, Z. B. Mahbub, K. R. Islam, M. S. Khan, A. Iqbal, N. Al-Emadi, M. B. I. Reaz, and M. T. Islam. Can AI help in screening viral and COVID-19 pneumonia? *IEEE Access*, 8:132665-132676, 2020. URL https://ieeexplore.ieee.org/document/9144185.

[9] T. Rahman, A. Khandakar, Y. Qiblawey, A. Tahir, S. Kiranyaz, S. B. A. Kashem, M. T. Islam, S. Al Maadeed, S. M. Zughaier, M. S. Khan, and M. E. H. Chowdhury. Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images. *Computers in Biology and Medicine*, 132:104319, 2021. URL https://doi.org/10.1016/j.compbiomed.2021.104319.

[10] A. Buslaev, V. I. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, and A. A. Kalinin. Albumentations: Fast and flexible image augmentations. *Information*, 11(2):125, 2020. URL https://doi.org/10.3390/info11020125.

## **License**

This project is licensed under the [MIT License](LICENSE).

