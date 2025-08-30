import os, random
import numpy as np
import librosa, librosa.display
import noisereduce as nr
import matplotlib.pyplot as plt
import pandas as pd
import IPython.display as ipd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Augmentation pipeline
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5)
])

# Emotion dictionary & labels
emotion_dict = {'ANG':0,'DIS':1,'FEA':2,'HAP':3,'NEU':4,'SAD':5}
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad']


# === Feature Extraction ===
def extract_features(file_path, do_aug=False):
    y, sr = librosa.load(file_path, sr=None)
    if do_aug:
        y = augment(samples=y, sample_rate=sr)
    y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    y = np.append(y[0], y[1:] - 0.97*y[:-1])
    intervals = librosa.effects.split(y, top_db=20)
    y = np.concatenate([y[s:e] for s,e in intervals])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)


# === Build CNN model ===
def build_model(input_len, n_classes=6):
    model = Sequential([
        Input(shape=(input_len,1)),
        Conv1D(64,3,activation='relu',padding='same'),
        MaxPooling1D(3,2,padding='same'),
        BatchNormalization(),
        Conv1D(128,3,activation='relu',padding='same'),
        MaxPooling1D(3,2,padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(128,activation='relu'), Dropout(0.3),
        Dense(n_classes,activation='softmax')
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


# === Training & Evaluation ===
def train_and_report(name, X, y, emotion_labels, results_path="results"):
    X_tr,X_tmp,y_tr,y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val,X_te,y_val,y_te = train_test_split(X_tmp,y_tmp,test_size=0.5,random_state=42,stratify=y_tmp)

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te  = scaler.transform(X_te)

    X_tr  = X_tr[...,np.newaxis]
    X_val = X_val[...,np.newaxis]
    X_te  = X_te[...,np.newaxis]

    y_tr  = to_categorical(y_tr,6)
    y_val = to_categorical(y_val,6)
    y_te  = to_categorical(y_te,6)

    model = build_model(input_len=X_tr.shape[1])
    cb = [ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=5,min_lr=1e-4),
          EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)]
    hist = model.fit(X_tr,y_tr,validation_data=(X_val,y_val),
                     epochs=50,batch_size=32,verbose=0,callbacks=cb)

    # Evaluate
    y_pred = model.predict(X_te)
    y_pred_cls = np.argmax(y_pred,axis=1)
    y_true_cls = np.argmax(y_te,axis=1)
    acc = accuracy_score(y_true_cls,y_pred_cls)
    print(f'=== {name} DATASET ===')
    print(f'Accuracy: {acc:.4f}')
    print(classification_report(y_true_cls,y_pred_cls,target_names=emotion_labels,digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true_cls,y_pred_cls)
    ConfusionMatrixDisplay(cm,display_labels=emotion_labels).plot(cmap=plt.cm.Blues,xticks_rotation=45)
    plt.title(f'Confusion Matrix – {name}')
    plt.savefig(os.path.join(results_path,f"confusion_{name.lower()}.png"))
    plt.close()

    # Plot Accuracy & Loss
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist.history['accuracy'], label='Train')
    plt.plot(hist.history['val_accuracy'], label='Val')
    plt.title(f'{name} Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'], label='Train')
    plt.plot(hist.history['val_loss'], label='Val')
    plt.title(f'{name} Loss')
    plt.legend()
    plt.savefig(os.path.join(results_path,f"training_{name.lower()}.png"))
    plt.close()

    # Save classification report
    report_txt = classification_report(y_true_cls,y_pred_cls,target_names=emotion_labels,digits=4)
    with open(os.path.join(results_path,f"classification_report_{name.lower()}.txt"),"w") as f:
        f.write(report_txt)

    return acc


# === Visualization: Sample Waveform & MFCC ===
def visualize_sample_by_emotion(files, emotion_dict, emotion_labels):
    files_by_emotion = {emo: [] for emo in emotion_dict}
    for f in files:
        emo_code = os.path.basename(f).split('_')[2].split('.')[0]
        if emo_code in emotion_dict:
            files_by_emotion[emo_code].append(f)

    for emo_code, emo_files in files_by_emotion.items():
        if not emo_files: continue
        file_path = random.choice(emo_files)

        emotion_name = emotion_labels[emotion_dict[emo_code]]
        filename = os.path.basename(file_path)

        print(f"\n================ {emotion_name} ================\n📄 File: {filename}")

        y_orig, sr = librosa.load(file_path, sr=None)
        y_aug = augment(samples=y_orig.copy(), sample_rate=sr)
        y_aug = nr.reduce_noise(y=y_aug, sr=sr, prop_decrease=0.8)

        # Waveform
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 2, 1)
        librosa.display.waveshow(y_orig, sr=sr)
        plt.title(f'Waveform - Original [{emotion_name}]')
        plt.subplot(1, 2, 2)
        librosa.display.waveshow(y_aug, sr=sr)
        plt.title('Waveform - Augmented')
        plt.tight_layout()
        plt.show()

        # MFCC
        mfcc_orig = librosa.feature.mfcc(y=y_orig, sr=sr, n_mfcc=13)
        mfcc_aug = librosa.feature.mfcc(y=y_aug, sr=sr, n_mfcc=13)

        plt.figure(figsize=(12, 3))
        plt.subplot(1, 2, 1)
        librosa.display.specshow(mfcc_orig, sr=sr, x_axis='time')
        plt.title('MFCC - Original')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        librosa.display.specshow(mfcc_aug, sr=sr, x_axis='time')
        plt.title('MFCC - Augmented')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        print("🔊 Original Audio:")
        display(ipd.Audio(y_orig, rate=sr))
        print("🔊 Augmented Audio:")
        display(ipd.Audio(y_aug, rate=sr))


# === Compare MFCC Original vs Augmented ===
def compare_mfcc_features_per_emotion(files, emotion_dict, emotion_labels):
    printed_emotions = set()
    for file_path in files:
        emo_code = os.path.basename(file_path).split('_')[2].split('.')[0]
        if emo_code in printed_emotions: continue
        printed_emotions.add(emo_code)

        emotion_name = emotion_labels[emotion_dict[emo_code]]
        filename = os.path.basename(file_path)

        print(f"\n================ {emotion_name} ================")
        print(f"📄 File: {filename}")

        features_orig = extract_features(file_path, do_aug=False)
        features_aug  = extract_features(file_path, do_aug=True)

        print("\n🎧 MFCC Features (Original):")
        print(np.round(features_orig, 3))
        print("\n🔊 MFCC Features (Augmented):")
        print(np.round(features_aug, 3))

        if len(printed_emotions) == len(emotion_dict):
            break
