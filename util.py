import os
import torch
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import io
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def plot_confusion_matrix(cm, class_names):
    figure, ax = plt.subplots(figsize=(len(class_names) + 2, len(class_names) + 2))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the plot to an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Convert the image to a PyTorch tensor
    image = Image.open(buf)
    image = transforms.ToTensor()(image)

    plt.close()

    return image


# 測試模型的函數
def model_test(model, classifier, test_loader, criterion, epoch, writer, image_type='day3', label_type='day3',
               test_type='Test'):
    model.eval().cuda()
    classifier.eval().cuda()
    test_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    all_logits = []
    criterion = torch.nn.CrossEntropyLoss()
    # Define your class names
    # class_names = ['Execellent', 'Good', 'Average', 'Poor', 'Fail']
    # class_names = ['Good', 'Average', 'Poor']
    class_names = ['Good & Average', 'Poor']
    with torch.no_grad():
        for batch_idx, (day3_images, day5_images, day5_labels, day3_labels) in enumerate(
                tqdm(test_loader, desc="Processing batches", position=0)):
            if image_type == 'day3':
                input_images = day3_images.to(device)
            elif image_type == 'day5':
                input_images = day5_images.to(device)

            if label_type == 'day3':
                input_label = day3_labels.to(device)
            elif label_type == 'day5':
                input_label = day5_labels.to(device)

            output_x4, output_x2, output = model(input_images)
            output, logits, _ = classifier(output_x4, output_x2, output)
            # input_label = torch.round(input_label).long()
            test_loss += criterion(output, input_label)

            # pred = output.argmax(dim=1, keepdim=True)
            _, pred_lables = torch.max(output, 1)

            # _, input_label = torch.max(input_label, 1)

            correct += pred_lables.eq(input_label).sum().item()

            # 收集所有真實標籤和預測標籤
            all_labels.extend(input_label.cpu().numpy())
            all_preds.extend(pred_lables.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    # 計算混淆矩陣
    # cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])
    # cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    # cm = np.reshape(cm, (1, 5, 5))
    cm_image = plot_confusion_matrix(cm, class_names)

    # 將混淆矩陣寫入 TensorBoard
    writer.add_image(f'{test_type}/Confusion Matrix', cm_image, global_step=epoch)

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    # 使用 t-SNE 將 logit 特徵降維到 2 維以便視覺化
    tsne = TSNE(n_components=2, random_state=42)
    logits_2d = tsne.fit_transform(all_logits)

    # 將降維後的特徵和對應的標籤寫入 TensorBoard
    writer.add_embedding(logits_2d, metadata=all_labels, global_step=epoch, tag=f'{test_type} Logits_Embedding')

    print(f'{test_type} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * accuracy:.2f}%)')

    # 將測試準確率寫入 TensorBoard
    writer.add_scalar(f'{test_type}/Accuracy', accuracy, epoch)

    return accuracy


def model_test_with_decoder(model, classifier, decoder, test_loader, epoch, writer,
                            image_type='day3', label_type='day3', test_type='Test'):
    model.eval().cuda()
    classifier.eval().cuda()
    decoder.eval().cuda()
    test_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    all_logits = []
    criterion = torch.nn.CrossEntropyLoss()
    # Define your class names
    # class_names = ['Execellent', 'Good', 'Average', 'Poor', 'Fail']
    # class_names = ['Good', 'Average', 'Poor']
    class_names = ['Good & Average', 'Poor']
    with torch.no_grad():
        for batch_idx, (day3_images, day5_images, day5_labels, day3_labels) in enumerate(
                tqdm(test_loader, desc="Processing batches", position=0)):
            if image_type == 'day3':
                input_images = day3_images.to(device)
            elif image_type == 'day5':
                input_images = day5_images.to(device)

            if label_type == 'day3':
                input_label = day3_labels.to(device)
            elif label_type == 'day5':
                input_label = day5_labels.to(device)

            output_x4, output_x2, output = model(input_images)
            decoder_image = decoder(output)
            # output_x4_day5, output_x2_day5, output_day5 = model(decoder_image)

            # output_x4, output_x2, output = torch.add(output_x4, output_x4_day5), \
            #                                torch.add(output_x2, output_x2_day5), \
            #                                torch.add(output, output_day5)

            input_images = torch.add(input_images, decoder_image)
            output_x4, output_x2, output = model(input_images)

            output, logits, _ = classifier(output_x4, output_x2, output)
            # input_label = torch.round(input_label).long()
            test_loss += criterion(output, input_label)

            # pred = output.argmax(dim=1, keepdim=True)
            _, pred_lables = torch.max(output, 1)

            # _, input_label = torch.max(input_label, 1)

            correct += pred_lables.eq(input_label).sum().item()

            # 收集所有真實標籤和預測標籤
            all_labels.extend(input_label.cpu().numpy())
            all_preds.extend(pred_lables.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    # 計算混淆矩陣
    # cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])
    # cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    # cm = np.reshape(cm, (1, 5, 5))
    cm_image = plot_confusion_matrix(cm, class_names)

    # 將混淆矩陣寫入 TensorBoard
    writer.add_image(f'{test_type}/Confusion Matrix', cm_image, global_step=epoch)

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    # 使用 t-SNE 將 logit 特徵降維到 2 維以便視覺化
    tsne = TSNE(n_components=2, random_state=42)
    logits_2d = tsne.fit_transform(all_logits)

    # 將降維後的特徵和對應的標籤寫入 TensorBoard
    writer.add_embedding(logits_2d, metadata=all_labels, global_step=epoch, tag=f'{test_type} Logits_Embedding')

    print(f'{test_type} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * accuracy:.2f}%)')

    # 將測試準確率寫入 TensorBoard
    writer.add_scalar(f'{test_type}/Accuracy', accuracy, epoch)

    return accuracy


class PairedImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, classes=5, label_smoothing=0.1):
        self.num_classes = classes
        self.data_dir = data_dir
        self.transform = transform
        self.day3_images = [os.path.join(data_dir, 'day3', img) for img in os.listdir(os.path.join(data_dir, 'day3'))]
        self.day5_images = [os.path.join(data_dir, 'day5', img) for img in os.listdir(os.path.join(data_dir, 'day5'))]
        self.class_mapping = {
            "Good": 0,
            "Average": 0,
            "Poor": 1,
        }
        self.label_smoothing = label_smoothing

    def label_smooth(self, one_hot_label):
        num_classes = len(one_hot_label)
        return (1.0 - self.label_smoothing) * one_hot_label + self.label_smoothing / num_classes

    def __len__(self):
        return min(len(self.day3_images), len(self.day5_images))

    def __getitem__(self, idx):
        day3_image_path = self.day3_images[idx]
        day5_image_path = self.day5_images[idx]

        day3_image = Image.open(day3_image_path)
        day5_image = Image.open(day5_image_path)

        # Extract class information from the file name
        day5_filename = os.path.basename(day5_image_path)
        day5_class = day5_filename.split('_')[-1].split('.')[0]
        day5_label = self.class_mapping.get(day5_class, -1)

        day3_filename = os.path.basename(day3_image_path)
        day3_class = day3_filename.split('_')[-1].split('.')[0]
        day3_label = self.class_mapping.get(day3_class, -1)

        # if day5_label != -1:
        #     day5_label_onehot = torch.zeros(self.num_classes)
        #     day5_label_onehot[day5_label] = 1
        #     day5_label_onehot = self.label_smooth(day5_label_onehot)
        #
        # if day3_label != -1:
        #     day3_label_onehot = torch.zeros(self.num_classes)
        #     day3_label_onehot[day3_label] = 1
        #     day3_label_onehot = self.label_smooth(day3_label_onehot)

        if self.transform:
            day3_image = self.transform(day3_image)
            day5_image = self.transform(day5_image)

        return day3_image, day5_image, day5_label, day3_label


class PairedImageDataset_test(Dataset):
    def __init__(self, data_dir, transform=None, classes=5, label_smoothing=0.1):
        self.num_classes = classes
        self.data_dir = data_dir
        self.transform = transform
        self.day3_images = [os.path.join(data_dir, 'day3', img) for img in os.listdir(os.path.join(data_dir, 'day3'))]
        self.day5_images = [os.path.join(data_dir, 'day5', img) for img in os.listdir(os.path.join(data_dir, 'day5'))]
        self.class_mapping = {
            "Good": 0,
            "Average": 0,
            "Poor": 1,
        }
        self.label_smoothing = label_smoothing

    def label_smooth(self, one_hot_label):
        num_classes = len(one_hot_label)
        return (1.0 - self.label_smoothing) * one_hot_label + self.label_smoothing / num_classes

    def __len__(self):
        return min(len(self.day3_images), len(self.day5_images))

    def __getitem__(self, idx):
        day3_image_path = self.day3_images[idx]
        day5_image_path = self.day5_images[idx]

        day3_image = Image.open(day3_image_path)
        day5_image = Image.open(day5_image_path)

        # Extract class information from the file name
        day5_filename = os.path.basename(day5_image_path)
        day5_class = day5_filename.split('_')[-1].split('.')[0]
        day5_label = self.class_mapping.get(day5_class, -1)

        day3_filename = os.path.basename(day3_image_path)
        day3_class = day3_filename.split('_')[-1].split('.')[0]
        day3_label = self.class_mapping.get(day3_class, -1)

        if self.transform:
            day3_image = self.transform(day3_image)
            day5_image = self.transform(day5_image)

        return day3_image, day5_image, day5_label, day3_label, day3_filename


# 保存EMA模型權重
def save_ema(ema, filepath):
    state_dict = {
        'ema_model': ema.ema_model.state_dict(),
        'step': ema.step,
        'initted': ema.initted,
    }
    torch.save(state_dict, filepath)

# 加载EMA模型權重
def load_ema(ema, filepath):
    state_dict = torch.load(filepath)
    ema.ema_model.load_state_dict(state_dict['ema_model'])
    ema.step = state_dict['step']
    ema.initted = state_dict['initted']
