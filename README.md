# 🛡️ Deepfake Detection & Prevention System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced AI-powered platform for detecting and preventing deepfake manipulations in digital media, ensuring trust and security in digital content through cutting-edge deep learning techniques.

## 🌟 Features

### 🔍 **Core Capabilities**
- **Real-Time Deepfake Detection** using MobileNetV2-based neural network
- **Live Webcam Analysis** for real-time monitoring and assessment
- **Batch Image Processing** with support for multiple image formats
- **Confidence Scoring** with detailed probabilistic analysis
- **Attention Heatmap Visualization** highlighting manipulated regions

### 🎨 **User Experience**
- **Professional Web Interface** built with Streamlit
- **Modern Dark Theme** with intuitive navigation
- **Interactive Dashboards** with real-time charts and graphs
- **Multi-page Application** structure for organized functionality
- **Responsive Design** optimized for different screen sizes

### 📊 **Analytics & Insights**
- **Model Performance Metrics** visualization
- **Training Progress Tracking** with loss and accuracy plots
- **Confusion Matrix Analysis** for detailed evaluation
- **Confidence Trend Analysis** for temporal assessment
- **Comprehensive Reporting** with detailed predictions

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher
pip package manager
Webcam (optional, for live demo)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/deepfake-detection-system.git
cd deepfake-detection-system
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Download or prepare your dataset**
```bash
# Create directory structure for training data
mkdir -p real_and_fake_face_detection/real_and_fake_face/training_real
mkdir -p real_and_fake_face_detection/real_and_fake_face/training_fake
```

### Usage

#### 🏃‍♂️ **Quick Demo (Web Application)**
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

#### 🎯 **Train Your Own Model**
```bash
python train.py
```

#### 🔮 **Make Predictions**
```bash
python predict.py
```

## 📁 Project Structure

```
deepfake-detection-system/
├── app.py                          # Main Streamlit web application
├── train.py                        # Model training script
├── predict.py                      # Prediction utilities
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── training_log.csv               # Training history and metrics
├── deepfake_detection_model.keras # Trained model (generated)
├── real_and_fake_face_detection/  # Dataset directory
│   └── real_and_fake_face/
│       ├── training_real/         # Real face images
│       └── training_fake/         # Fake face images
└── plots/                         # Generated visualization plots
    ├── accuracy_plot.png
    ├── loss_plot.png
    └── confusion_matrix.png
```

## 🧠 Technical Architecture

### **Deep Learning Model**
- **Base Architecture**: MobileNetV2 (ImageNet pre-trained)
- **Transfer Learning**: Fine-tuned for deepfake detection
- **Input Resolution**: 96x96 RGB images
- **Classification**: Binary (Real vs Fake)
- **Optimization**: Adam optimizer with adaptive learning rate

### **Model Architecture Details**
```
MobileNetV2 (Feature Extractor)
├── GlobalAveragePooling2D
├── Dense(512) + ReLU + BatchNormalization + Dropout(0.5)
├── Dense(256) + ReLU + BatchNormalization + Dropout(0.3)
└── Dense(2) + Softmax (Binary Classification)
```

### **Training Configuration**
- **Data Augmentation**: Rotation, Zoom, Shear, Brightness adjustment
- **Regularization**: Dropout, BatchNormalization, Class weights
- **Validation Split**: 20%
- **Early Stopping**: Patience-based with best weights restoration
- **Learning Rate Scheduling**: Automatic reduction on plateau

## 📊 Performance Metrics

### **Model Performance**
- **Training Accuracy**: 79.66%
- **Validation Accuracy**: 54.88%
- **Architecture**: MobileNetV2 + Custom Classification Head
- **Training Epochs**: 17 (with early stopping)

### **Key Features**
- **Overfitting Prevention**: Multiple regularization techniques
- **Class Imbalance Handling**: Weighted loss function
- **Real-time Inference**: Optimized for live webcam processing
- **Batch Processing**: Efficient multi-image analysis

## 🎯 Use Cases

### **Primary Applications**
- **🔍 Digital Forensics**: Law enforcement and legal proceedings
- **📰 Media Verification**: News organizations and fact-checking
- **🌐 Social Media Monitoring**: Content moderation and authenticity
- **🏢 Corporate Security**: Employee verification and fraud prevention
- **🎓 Academic Research**: Deepfake detection methodology studies

### **Industry Impact**
- **Cybersecurity**: Proactive defense against digital manipulation
- **Media Integrity**: Maintaining trust in digital communications
- **Legal Technology**: Evidence authentication and verification

## 🛠️ API Reference

### **Core Functions**

#### `preprocess_image(image, target_size=(96, 96))`
Preprocesses input image for model prediction.
- **Parameters**: `image` (numpy array), `target_size` (tuple)
- **Returns**: Normalized image array

#### `predict_image(image)`
Performs deepfake detection on single image.
- **Parameters**: `image` (numpy array)
- **Returns**: `result` (str), `confidence` (float), `heatmap` (numpy array)

#### `batch_predict(directory)`
Processes multiple images in a directory.
- **Parameters**: `directory` (str)
- **Returns**: Dictionary of results with filenames as keys

## 📋 Requirements

```txt
streamlit>=1.28.0
tensorflow>=2.10.0
opencv-python>=4.7.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
scikit-learn>=1.1.0
Pillow>=9.2.0
```

## 🚀 Advanced Usage

### **Custom Training**
```python
# Modify training parameters in train.py
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
VALIDATION_SPLIT = 0.2
```

### **Web Application Customization**
```python
# Customize UI theme in app.py
# Modify CSS styles in the st.markdown() sections
# Add new pages by extending the navigation logic
```

### **Batch Processing Example**
```python
from predict import batch_predict

# Process all images in a directory
results = batch_predict("path/to/images/")
for filename, (prediction, confidence) in results.items():
    print(f"{filename}: {prediction} ({confidence:.2f}%)")
```

## 🔧 Configuration

### **Model Training Parameters**
```python
# Data Augmentation Settings
ROTATION_RANGE = 30
ZOOM_RANGE = 0.3
BRIGHTNESS_RANGE = [0.7, 1.3]
HORIZONTAL_FLIP = True

# Training Settings
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 8
```

### **Web Application Settings**
```python
# UI Configuration
THEME = "dark"
PAGE_TITLE = "Deepfake Detection System"
SIDEBAR_WIDTH = 300
WEBCAM_RESOLUTION = (640, 480)
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### **Development Guidelines**
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team** for the excellent deep learning framework
- **Streamlit Team** for the intuitive web application framework
- **MobileNetV2 Authors** for the efficient CNN architecture
- **Open Source Community** for various tools and libraries used

## 📞 Contact & Support

### **Developer**
- **Name**: Pratham Katariya
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Email**: your.email@example.com

### **Issues & Bug Reports**
Please use the [GitHub Issues](https://github.com/yourusername/deepfake-detection-system/issues) page for:
- Bug reports
- Feature requests
- Technical support
- General questions

## 🔄 Version History

### **v1.0.0** (Current)
- Initial release with core deepfake detection functionality
- Streamlit web application with professional UI
- Real-time webcam analysis
- Batch processing capabilities
- Model training and evaluation scripts

### **Upcoming Features**
- [ ] Video deepfake detection
- [ ] REST API development
- [ ] Mobile app support
- [ ] Cloud deployment options
- [ ] Enhanced model architectures

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

*Built with ❤️ for digital media security and authenticity*

</div>
