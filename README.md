# CAP 4630 Final Project: Smart Waste Sorting Assistant
An AI-powered web application that classifies waste images into material categories and provides disposal guidance.  
Built with TensorFlow/Keras (MobileNetV2 transfer learning) and deployed with Streamlit.
---
## Overview
This project develops an intelligent image classification system for waste sorting from images.  
The system supports end-to-end workflow:
- Dataset preparation and cleaning
- Model training with transfer learning
- Evaluation with confusion matrix and macro metrics
- Interactive Streamlit demo app for real-time prediction
The final 5-class model achieves:
- **Accuracy:** 77.51%
- **Macro-F1:** 0.7781
---
## Project Objectives
- Automatically classify waste images with practical accuracy
- Distinguish among key waste categories for disposal decisions
- Provide confidence-aware predictions in a user-friendly web app
- Build a reproducible training + evaluation pipeline for CAP 4630
---
## Dataset
- **Source:** Waste classification dataset from Kaggle  
  https://www.kaggle.com/datasets/phenomsg/waste-classification
- **Image type:** RGB photos
- **Input size:** Standardized to **224 × 224** during training/inference
### Final Training Classes (5-class model)
- `plastic`
- `paper`
- `glass`
- `metal`
- `organic`
### Current Class Counts (`data/raw_5class`)
- plastic: 350  
- paper: 149  
- glass: 213  
- metal: 349  
- organic: 294  
- **Total:** 1355 images
Uncertain images were isolated into `data/review_hold` (not used for final training).
---
## Model Architecture
The project uses **MobileNetV2 Transfer Learning**:
- **Backbone:** MobileNetV2 pre-trained on ImageNet
- **Input:** 224×224×3
- **Custom head:** GlobalAveragePooling2D + Dropout + Dense softmax
- **Loss:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Regularization/robustness:** augmentation + class weighting + callbacks
### Data Augmentation
- Random horizontal flip
- Random rotation
- Random zoom
- Random contrast
---
## Performance Metrics
### Final 5-Class Model (Epoch 12)
- **Accuracy:** 0.7751
- **Macro Precision:** 0.7822
- **Macro Recall:** 0.7755
- **Macro F1:** 0.7781
### Key Evaluation Artifacts
- `reports/metrics_5class_e12.json`
- `reports/cm_5class_e12.png`
---
## Project Structure
```text
cap4630-smart-waste-sorting/
├── app/
│   └── streamlit_app.py             # Streamlit inference app
├── src/
│   ├── train.py                     # Model training pipeline
│   └── evaluate.py                  # Evaluation + metrics + confusion matrix
├── data/
│   ├── raw_5class/                  # Final 5-class training data
│   ├── raw/                         # Earlier 6-class/working data
│   └── review_hold/                 # Uncertain samples excluded from training
├── models/
│   ├── waste_classifier.keras       # App-loaded model
│   ├── class_names.json             # App-loaded class map
│   ├── waste_classifier_5class.keras
│   └── class_names_5class.json
├── reports/
│   ├── metrics_5class_e12.json
│   ├── cm_5class_e12.png
│   ├── e8_metrics.json
│   ├── e12_metrics.json
│   └── e15_metrics.json
├── requirements.txt
└── .gitignore
Installation
Prerequisites
Python 3.10+ (3.11 recommended)
pip
Windows/macOS/Linux
Setup
# 1) Clone repository
git clone <your-repo-url>
cd cap4630-smart-waste-sorting
# 2) Create virtual environment
python -m venv .venv
# 3) Activate virtual environment
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
# 4) Install dependencies
pip install -r requirements.txt
Usage
1) Train
Final 5-class run (recommended)
python src/train.py --data-dir data/raw_5class --epochs 12 --img-size 224 --batch-size 32 --learning-rate 0.0005 --output-model models/waste_classifier_5class.keras --output-class-map models/class_names_5class.json
2) Evaluate
python src/evaluate.py --data-dir data/raw_5class --model-path models/waste_classifier_5class.keras --class-map models/class_names_5class.json --metrics-out reports/metrics_5class_e12.json --cm-out reports/cm_5class_e12.png
3) Set model for app
# Windows PowerShell
Copy-Item "models\waste_classifier_5class.keras" "models\waste_classifier.keras" -Force
Copy-Item "models\class_names_5class.json" "models\class_names.json" -Force
4) Run demo app
streamlit run app/streamlit_app.py
Key Features
Transfer learning with MobileNetV2
End-to-end training/evaluation pipeline
Class weighting for imbalance handling
Confusion matrix + macro metric reporting
Streamlit web app for interactive predictions
Visual, demo-ready interface and probability outputs
Technical Notes
The app is a web app (Streamlit), not a static website.
Inference preprocessing is aligned with training pipeline.
Model quality is sensitive to label quality and class balance.
For ambiguous images, human review is recommended.
Limitations
Performance depends on dataset quality and representativeness
Ambiguous/multi-object images can reduce prediction reliability
Domain shift (new backgrounds/lighting) may impact accuracy
Future Improvements
Expand and rebalance dataset per class
Improve hard-case handling for ambiguous items
Add explicit “unknown/manual review” fallback strategy
Explore fine-tuning unfreezed layers after warm-up
Optional mobile deployment via TensorFlow Lite
Course Context
This project was developed for:

Course: CAP 4630 – Introduction to Artificial Intelligence
Final deliverables: codebase, presentation, and recorded demo
References
Plant/waste classification datasets from Kaggle (as cited in presentation)
TensorFlow/Keras documentation
MobileNetV2 transfer learning references
License
Add your preferred license here (e.g., MIT).

Contact
Name: Anthony Longo
GitHub: <your-github-profile>
Email: <your-email>
