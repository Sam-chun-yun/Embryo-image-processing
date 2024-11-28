import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.model import EFE_subnetwork, ECP_subnetwork_logit, Decoder
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import matplotlib
import torch.nn.functional as F
from util import PairedImageDataset_test
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')

transform_test = transforms.Compose([
    transforms.Resize((352, 352)),  # 調整圖像大小
    transforms.ToTensor(),  # 將圖像轉換為張量
    transforms.Normalize(mean=[0.485], std=[0.229])  # 歸一化（單通道圖像）
])

# setting parameters
initial_ch = 32
# 資料集類別
num_classes = 2

batch_size = 1

# 測試樣本路徑
test_dir = r'test_data'

test_dataset = PairedImageDataset_test(test_dir, transform=transform_test, classes=num_classes)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

device = 'cpu'

# model 實例化
EFE = EFE_subnetwork(initial_ch)
ECP = ECP_subnetwork_logit(initial_ch * (2 ** 5), num_class=2)
feature_decoder = Decoder(initial_ch * (2 ** 5))

#  weight path setting
weight_path = r'weight'

#  EFE_subnetwork weight loading
path = os.path.join(weight_path, 'EFE_subnetwork.pth')
state_dict = torch.load(path, map_location=device)
EFE.load_state_dict(state_dict)

#  ECP_subnetwork weight loading
path = os.path.join(weight_path, 'ECP_subnetwork.pth')
state_dict = torch.load(path, map_location=device)
ECP.load_state_dict(state_dict)

#  feature dilation module weight loading
path = os.path.join(weight_path, 'feature_decoder.pth')
state_dict = torch.load(path, map_location=device)
feature_decoder.load_state_dict(state_dict)

class CombinedModel(nn.Module):
    def __init__(self, efe_model, ecp_model, decoder):
        super(CombinedModel, self).__init__()
        self.efe_model = efe_model
        self.ecp_model = ecp_model
        self.decoder = decoder

    def forward(self, x):
        output_x4, output_x2, output = self.efe_model(x)
        decoder_image = self.decoder(output)
        input_images = torch.add(x, decoder_image)
        output_x4, output_x2, output = self.efe_model(input_images)
        ecp_output, _, _ = self.ecp_model(output_x4, output_x2, output)

        return ecp_output, input_images



def visual_test():
    # save path
    save_dir = "CAM_output_images_FEM_0726"
    # Wrap EFE and ECP in the CombinedModel
    model = CombinedModel(EFE, ECP, feature_decoder)
    model.eval()
    # Define the target layer for GradCAM

    target_layers = [model.efe_model.FEM[4]]

    for batch_idx, (day3_images, day5_images, day5_labels, day3_labels, day3_filenames) in enumerate(test_dataloader):
        # Construct the GradCAM object
        cam = GradCAM(model=model, target_layers=target_layers)
        pred, input_image_np = model(day3_images)
        groud_true = day5_labels

        input_image_np = input_image_np.cpu()
        input_image_np = input_image_np.detach().numpy()


        # day3_image = day3_images[0].unsqueeze(0)
        pred_score = F.softmax(pred[0])
        pred_score = (pred_score.max() * 100).detach().numpy().item()
        pred_score = round(pred_score, 2)

        _, pred = torch.max(pred[0], 0)

        input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())

        input_image_np = np.array(input_image_np).squeeze().transpose((1, 2, 0))

        # Specify the target category (here, category 2)
        targets = [ClassifierOutputTarget(pred)]

        # Generate the CAM for the input image
        grayscale_cam = cam(input_tensor=day3_images, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(input_image_np, grayscale_cam, use_rgb=True)

        class_names = ['Average', 'Poor']
        level = class_names[pred]
        groud_true_level = class_names[groud_true]

        # Save the original image and the CAM overlay
        for i, filename in enumerate(day3_filenames):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(input_image_np)
            plt.title("Original Image")

            plt.subplot(1, 2, 2)
            plt.imshow(visualization)
            plt.title(f"Grad-CAM Overlay\nTrue rating: {groud_true_level}\nPredicted rating: {level}"
                      f"\nScore: {pred_score}")


            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{os.path.splitext(filename)[0]}_gradcam.png")
            plt.savefig(save_path)
            plt.close()

if __name__ == '__main__':
    visual_test()
