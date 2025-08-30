# 🎤 Speech Emotion Recognition with CNN and Data Augmentation

This repository contains the implementation of my undergraduate thesis project on **Speech Emotion Recognition (SER)** using **Convolutional Neural Networks (CNNs)** with **MFCC features**.  
The experiments are conducted on the **CREMA-D dataset**, with three variations of training data:

- **Original Dataset** (raw data only)
- **Augmented Dataset** (audio augmented using audiomentations)
- **Combined Dataset** (original + augmented)

The goal of this project is to evaluate the effect of **data augmentation** on emotion classification performance.

---

## 📂 Repository Structure

speech-emotion-recognition/
│
├── data/ # CREMA-D .wav files (flat directory)
│
├── results/ # all evaluation outputs
│ ├── accuracy_original.png
│ ├── loss_original.png
│ ├── confusion_original.png
│ ├── classification_report_original.txt
│ ├── accuracy_augmented.png
│ ├── loss_augmented.png
│ ├── confusion_augmented.png
│ ├── classification_report_augmented.txt
│ ├── accuracy_combined.png
│ ├── loss_combined.png
│ ├── confusion_combined.png
│ └── classification_report_combined.txt
│
├── src/
│ ├── utils.py # feature extraction, augmentations, model building, plotting
│ └── main.py # run experiments (Original, Augmented, Combined)
│
├── requirements.txt # dependencies
└── README.md # documentation

---

## ⚙️ Installation

Run the following inside Google Colab (recommended) or a local environment:

```bash
pip install numpy pandas librosa noisereduce audiomentations tensorflow scikit-learn matplotlib

---

## ▶️ Usage

1. Place the CREMA-D dataset (.wav files) inside the data/ directory.
Example filename: 1001_DFA_ANG_XX.wav

2. Run the training and evaluation script:

cd src
python main.py

3. Results will be saved inside the results/ folder:

Training accuracy & loss plots

Confusion matrix plots

Classification reports (per dataset)

## 📊 Example Results
Original Dataset

Accuracy: ~44.8%

Augmented Dataset

Accuracy: ~38.6%

Combined Dataset

Accuracy: ~46.4%

Confusion matrices and full classification reports are available in the results/ directory.

## 📑 Dataset

This project uses the [CREMA-D Dataset](https://www.kaggle.com/datasets/ejlok1/cremad) from Kaggle.

- The dataset contains 7,442 audio clips (.wav) of 91 actors.
- Each clip is labeled with one of six emotions: Angry, Disgust, Fear, Happy, Neutral, Sad.
- Files are named following the pattern: `ActorID_Emotion_Sentence.wav`.

⚠️ Due to size constraints, the dataset is **not included** in this repository.  
Please download it manually from Kaggle and place the `.wav` files into:



## ✨ Key Features

Audio preprocessing with noise reduction

Data augmentation with audiomentations (time stretch, pitch shift, Gaussian noise, shift)

Feature extraction using MFCC

CNN architecture with Conv1D layers

Comparative evaluation of Original, Augmented, and Combined datasets

## 📌 Notes

This project does not save trained models (.h5) or perform inference.

The focus is on training & evaluation for research purposes only.

All experiments are reproducible via main.py.

## 🧑‍💻 Author

Dede Septa Maulana Fajar
Undergraduate Thesis Project – 2025
