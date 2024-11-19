# Diabetic Retinopathy Detection Using CNNs

## Introduction  
Diabetic Retinopathy (DR) is a leading cause of vision impairment in individuals with long-standing diabetes. Early detection and timely treatment can prevent vision loss and improve patient outcomes. However, manual grading of retinal images is time-consuming and error-prone. This project aims to develop automated, reliable, and accurate deep learning models to detect and grade Diabetic Retinopathy, facilitating early intervention and personalized treatment.

## Dataset  
The dataset consists of a collection of high-resolution retinal images captured under various imaging conditions. Each image is annotated by medical professionals with the presence or absence of Diabetic Retinopathy:

- **0**: Diabetic Retinopathy  
- **1**: No Diabetic Retinopathy  

### Dataset Highlights:
- High-resolution retinal images.
- Ground truth labels provided by medical professionals.
- Balanced dataset with equal distribution of classes.

---

## Model Architectures  

This project implements two deep learning models using convolutional neural networks (CNNs):  
### **Model 1: VGG16-Based Architecture**  
Utilizes a pre-trained VGG16 as a feature extractor, enhanced with:  
- **Multi-Head Attention**: For learning spatial relationships.  
- **Gaussian Noise**: To improve robustness.  
- Fully connected layers with dropout for generalization.

#### Results:  
- **Accuracy**: 95%  
- **Precision**: 94% - 96%  
- **Recall**: 94% - 96%  
- **F1-Score**: 95%

---

### **Model 2: MobileNetV2-Based Lightweight Architecture**  
Utilizes a pre-trained MobileNetV2 for computational efficiency, enhanced with:  
- **Multi-Head Attention**: Simplified with fewer heads to balance efficiency and accuracy.  
- **Gaussian Noise**: To enhance robustness.  
- Compact dense layers with higher dropout for generalization.

#### Results:  
- **Accuracy**: 98%  
- **Precision**: 97% - 99%  
- **Recall**: 97% - 99%  
- **F1-Score**: 98%

---

## Usage  

### Requirements  
- Python 3.7+  
- TensorFlow 2.x  
- Required libraries: `keras`, `numpy`, `pandas`, `matplotlib`  

Install dependencies:  
```bash
pip install -r requirements.txt
```

### Training the Models  
To train Model 1 (VGG16-based):  
```python
python train_vgg16_model.py
```

To train Model 2 (MobileNetV2-based):  
```python
python train_mobilenetv2_model.py
```

### Evaluate Models  
Run the evaluation script to generate classification metrics:  
```python
python evaluate_model.py
```

---

## Results and Analysis  

| Metric       | Model 1 (VGG16) | Model 2 (MobileNetV2) |  
|--------------|------------------|-----------------------|  
| Accuracy     | 95%             | 98%                   |  
| Precision    | 94% - 96%       | 97% - 99%             |  
| Recall       | 94% - 96%       | 97% - 99%             |  
| F1-Score     | 95%             | 98%                   |  

---

## Key Features  

- **Attention Mechanisms**: Integrated attention layers enhance feature representation by focusing on critical areas in retinal images.  
- **Transfer Learning**: Efficient use of pre-trained weights to expedite convergence.  
- **Robustness**: Gaussian noise improves model generalization under diverse imaging conditions.  

---

## Future Work  
- Extend the dataset for multi-class classification of DR severity levels.  
- Fine-tune models on domain-specific data for improved performance.  
- Deploy as a web-based tool for real-time screening.  

---

## Acknowledgments  
Special thanks to healthcare professionals who contributed to the dataset annotations.

Feel free to replace placeholders like `train_vgg16_model.py` or `evaluate_model.py` with your actual filenames. Let me know if you want me to refine further!
