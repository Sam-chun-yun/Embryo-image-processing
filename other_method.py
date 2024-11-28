import os
from sklearn import naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score, confusion_matrix
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Data paths
train_day3_path = 'data/train/day3'
train_day5_path = 'data/train/day5'
test_day3_path = 'data/test/day3'
test_day5_path = 'data/test/day5'

# Transforms
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

# VGG model
vgg_model = models.vgg16(pretrained=True).eval().cuda()

def get_image_paths(base_path):
    return sorted([os.path.join(base_path, file_name) for file_name in os.listdir(base_path)])

def get_labels(file_names):
    return [name.split('_')[9].split('.jpg')[0] for name in file_names]

def load_images(image_paths):
    images = []
    for path in image_paths:
        image = Image.open(path)
        image = transform(image).unsqueeze(0).cuda()
        images.append(image)
    return images

def extract_features(images):
    features = []
    for image in images:
        with torch.no_grad():
            output = vgg_model.features(image).view(-1)
            features.append(output)
    features = torch.stack(features)
    return (features - features.min()) / (features.max() - features.min())

def label_to_numeric(labels):
    numeric_labels = []
    for label in labels:
        if label in ['Poor', 'POOR']:
            numeric_labels.append(0)
        elif label in ['Average', 'AVERAGE', 'Good', 'GOOD']:
            numeric_labels.append(1)
        else:
            numeric_labels.append(2)
    return numeric_labels

# Load data
train_day3_names = get_image_paths(train_day3_path)
train_day5_names = get_image_paths(train_day5_path)
test_day3_names = get_image_paths(test_day3_path)
test_day5_names = get_image_paths(test_day5_path)

train_day3_images = load_images(train_day3_names)
test_day3_images = load_images(test_day3_names)

train_features = extract_features(train_day3_images).cpu().numpy()
test_features = extract_features(test_day3_images).cpu().numpy()

train_labels = label_to_numeric(get_labels(train_day5_names))
test_labels = label_to_numeric(get_labels(test_day5_names))

# Classifiers
classifiers = {
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Naive Bayes': naive_bayes.MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'XGBoost': XGBClassifier(),
    'Random Forest': RandomForestClassifier()
}

predictions = {}
metrics = {}

for name, clf in classifiers.items():
    clf.fit(train_features, train_labels)
    preds = clf.predict(test_features)
    predictions[name] = preds

    metrics[name] = {
        'precision_macro': precision_score(test_labels, preds, average='macro', zero_division=0),
        'recall_macro': recall_score(test_labels, preds, average='macro'),
        'jaccard_macro': jaccard_score(test_labels, preds, average='macro'),
        'precision_weighted': precision_score(test_labels, preds, average='weighted', zero_division=0),
        'recall_weighted': recall_score(test_labels, preds, average='weighted'),
        'jaccard_weighted': jaccard_score(test_labels, preds, average='weighted'),
        'precision_micro': precision_score(test_labels, preds, average='micro', zero_division=0),
        'recall_micro': recall_score(test_labels, preds, average='micro'),
        'jaccard_micro': jaccard_score(test_labels, preds, average='micro'),
        'accuracy': accuracy_score(test_labels, preds),
        'confusion_matrix': confusion_matrix(test_labels, preds)
    }

# Print metrics
for name, metric in metrics.items():
    print(f"=== {name} ===")
    for metric_name, score in metric.items():
        if metric_name == 'confusion_matrix':
            print(f"{metric_name}:\n{score}\n")
        else:
            print(f"{metric_name}: {round(score, 4)}")

# Visualization of confusion matrices
for name, metric in metrics.items():
    plt.figure(figsize=(8, 6))
    plt.title(f"Confusion Matrix: {name}")
    plt.imshow(metric['confusion_matrix'], cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
