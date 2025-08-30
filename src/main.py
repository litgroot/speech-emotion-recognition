import os
import numpy as np
from utils import extract_features, train_and_report, visualize_sample_by_emotion, compare_mfcc_features_per_emotion, emotion_dict, emotion_labels

# Dataset path (update sesuai lokasi dataset)
dataset_path = "../data/"

# Load dataset
files = [os.path.join(dataset_path,f) for f in os.listdir(dataset_path) if f.endswith('.wav')]
files.sort()

# Extract features
feat_orig = np.vstack([extract_features(p, False) for p in files])
feat_aug  = np.vstack([extract_features(p, True)  for p in files])
lab_orig = np.array([emotion_dict[os.path.basename(p).split('_')[2].split('.')[0]] for p in files])
lab_aug  = lab_orig.copy()

X_orig, y_orig = feat_orig, lab_orig
X_aug , y_aug  = feat_aug , lab_aug
X_comb = np.vstack([X_orig,X_aug])
y_comb = np.hstack([y_orig,y_aug])

datasets = {
    'Original': (X_orig,y_orig),
    'Augmented': (X_aug,y_aug),
    'Combined': (X_comb,y_comb)
}

# Create results dir
results_path = "../results"
os.makedirs(results_path, exist_ok=True)

# Train & evaluate
for name,(X,y) in datasets.items():
    train_and_report(name,X,y,emotion_labels,results_path)

# Visualization
visualize_sample_by_emotion(files, emotion_dict, emotion_labels)
compare_mfcc_features_per_emotion(files, emotion_dict, emotion_labels)
