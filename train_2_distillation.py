import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from models.model import EFE_subnetwork, FM_subnetwork, init_net, set_requires_grad, \
    ECP_subnetwork_logit, Decoder
import itertools
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
from ema_pytorch import EMA
from losses import wgan_gp_loss
from util import model_test, PairedImageDataset, model_test_with_decoder, save_ema
from trainer import trainer_cls, trainer_extractor_with_decoder, \
    trainer_2_extractor_dis, trainer_cls_with_decoder

# 自訂影像增強變換
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 隨機翻轉
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),  # 隨機旋轉
    transforms.RandomApply([
        transforms.RandomChoice([
            transforms.RandomResizedCrop(352, scale=(0.8, 1.0), ratio=(0.8, 1.0)),  # 隨機裁剪和縮放
            transforms.RandomAdjustSharpness(1),
            transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
            transforms.RandomAffine(degrees=0, shear=(-0.5, 0.5))
        ])
    ], p=0.5),
    transforms.Resize((352, 352)),  # 調整圖像大小
    transforms.ToTensor(),  # 將圖像轉換為張量
    transforms.Normalize(mean=[0.485], std=[0.229])  # 歸一化（單通道圖像）
])

transform_test = transforms.Compose([
    transforms.Resize((352, 352)),  # 調整圖像大小
    transforms.ToTensor(),  # 將圖像轉換為張量
    transforms.Normalize(mean=[0.485], std=[0.229])  # 歸一化（單通道圖像）
])

# setting parameters
# 網路初始層通道數
initial_ch = 32

# 自訂資料集類別
num_classes = 2

# 訓練時batch_size設置
batch_size = 32

# 引導訓練batch_size設置
GL_batch_size = 24

weight_initial_type = 'kaiming'

# 使用GPU數量，如果使用兩塊GPU則設置為'0,1'，以此類推
gpu_id = '0'
gpu_id = [int(id) for id in gpu_id.split(',')]

# 請替換為實際的數據目錄路徑
data_dir = r'data/train'
test_dir = r'data/test'

# 定義自訂數據集，將day3和day5圖像與類別標籤配對起來
paired_dataset = PairedImageDataset(data_dir, transform=transform, classes=num_classes)
test_dataset = PairedImageDataset(test_dir, transform=transform_test, classes=num_classes)
train_dataset = PairedImageDataset(data_dir, transform=transform_test, classes=num_classes)

# 創建數據加載器，指定批次大小和是否要打亂數據
dataloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
dataloader_gan = DataLoader(paired_dataset, batch_size=GL_batch_size, shuffle=True, num_workers=4, pin_memory=True)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

# 修改模型的輸出層以適應你的分類任務
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AMP初始化
scaler = GradScaler()

from torch.optim.lr_scheduler import StepLR

total_batches = len(dataloader)

# 定義損失函數
criterion = torch.nn.CrossEntropyLoss()  # 使用標準交叉熵損失


def train(image_type='day3', label_type='day3', writer='writer', path='path'):
    # model 實例化
    EFE = EFE_subnetwork(initial_ch)                                    # Day3 EFE subnetwork 初始化
    EFE_day5 = EFE_subnetwork(initial_ch)                               # Day5 EFE subnetwork 初始化
    FM = FM_subnetwork(initial_ch * (2 ** 3))                           # FM subnetwork 初始化
    ECP = ECP_subnetwork_logit(initial_ch * (2 ** 5), num_class=2)      # ECP subnetwork 初始化
    feature_decoder = Decoder(initial_ch * (2 ** 5))                    # feature dilation module 初始化

    # 設置是否導入預訓練權重
    load_weight = True
    weight_path = 'pretrain_weight'

    if load_weight is True:
        path_efe = os.path.join(weight_path, 'EFE_day5_subnetwork_last.pth')
        state_dict = torch.load(path_efe, map_location=device)
        EFE_day5.load_state_dict(state_dict)

        path_efe = os.path.join(weight_path, 'ECP_subnetwork_last.pth')
        state_dict = torch.load(path_efe, map_location=device)
        ECP.load_state_dict(state_dict)

    # 初始化權重與平行分配多GPU
    EFE = init_net(EFE, init_type=weight_initial_type, gpu_ids=gpu_id, load_weight=False)
    EFE_day5 = init_net(EFE_day5, init_type=weight_initial_type, gpu_ids=gpu_id, load_weight=True)
    FM = init_net(FM, init_type=weight_initial_type, gpu_ids=gpu_id, load_weight=False)
    ECP = init_net(ECP, init_type=weight_initial_type, gpu_ids=gpu_id, load_weight=True)
    feature_decoder = init_net(feature_decoder, init_type=weight_initial_type, gpu_ids=gpu_id, load_weight=False)

    cls_optimizer_day5 = torch.optim.Adam(itertools.chain(EFE_day5.parameters(), ECP.parameters()), lr=2e-4,
                                          weight_decay=4e-5)

    cls_optimizer_day3 = torch.optim.Adam(itertools.chain(EFE.parameters(), ECP.parameters()), lr=2e-4,
                                          weight_decay=4e-5)

    FM_optimizer = torch.optim.RMSprop(FM.parameters(), lr=3e-6)
    EFE_optimizer_with_decoder = torch.optim.RMSprop(itertools.chain(EFE.parameters(), feature_decoder.parameters()),
                                                     lr=2e-4)

    ema_efe = EMA(
        EFE,
        beta=0.995,  # exponential moving average factor
        update_after_step=10,  # only after this number of .update() calls will it start updating
        update_every=1,  # how often to actually update, to save on compute (updates every 10th .update() call)
    )

    ema_efe_day5 = EMA(
        EFE_day5,
        beta=0.995,  # exponential moving average factor
        update_after_step=10,  # only after this number of .update() calls will it start updating
        update_every=1,  # how often to actually update, to save on compute (updates every 10th .update() call)
    )

    ema_ecp = EMA(
        ECP,
        beta=0.995,  # exponential moving average factor
        update_after_step=10,  # only after this number of .update() calls will it start updating
        update_every=1,  # how often to actually update, to save on compute (updates every 10th .update() call)
    )

    scheduler = StepLR(EFE_optimizer_with_decoder, step_size=15, gamma=0.4)
    scheduler_day3 = StepLR(cls_optimizer_day3, step_size=15, gamma=0.4)
    scheduler_day5 = StepLR(cls_optimizer_day5, step_size=10, gamma=0.4)
    scheduler_FM = StepLR(FM_optimizer, step_size=30, gamma=0.4)
    # scheduler_EFE = StepLR(EFE_optimizer_with_decoder, step_size=240, gamma=0.4)

    # FM_scheduler = StepLR(FM_optimizer, step_size=20, gamma=0.2)
    # EFE_scheduler = StepLR(EFE_optimizer, step_size=20, gamma=0.2)

    # 訓練切換部署設置
    step1 = 0                   # 訓練day5_EFE與ECP subnetwork
    step2 = step1 + 0           # 訓練day3_EFE與ECP subnetwork (不須設置)
    step3 = step2 + 200         # 訓練FM subnetwork
    num_epochs = step3 + 5      # 透過day5_EFE與FM subnetwork引導訓練day3_EFE subnetwork

    # 訓練各部分調適次數
    loop_s = 40
    s3 = 2                      # 重複訓練FM subnetwork次數，使其收斂至所設閥值
    s5 = 2                      # 訓練利用day3 EFE subnetwork進行預測次數

    temp_accuracy = 0.0
    temp_accuracy_d3 = 0.0
    efe_iter = 0
    gp_value = 0.0
    accuracy = 0.0
    threshold = 1.0

    # 訓練循
    for epoch in range(num_epochs):
        # 訓練day5_EFE與ECP subnetwork
        if epoch < step1:
            loss, loss_cl, loss_focal, loss_supcon, loss_bitemp = \
                trainer_cls(extractor=EFE_day5, classifier=ECP, classifier_optimizer=cls_optimizer_day5,
                            dataloader=dataloader, ema_extor=ema_efe_day5, ema_clsfer=ema_ecp,
                            image_type='day3', device=device)

            avg_loss = (loss / batch_size)
            avg_loss_cl = (loss_cl / batch_size)
            avg_loss_focal = (loss_focal / batch_size)
            avg_loss_supcon = (loss_supcon / batch_size)
            avg_loss_bitemp = (loss_bitemp / batch_size)
            scheduler_day5.step()

            learning_rate = cls_optimizer_day5.param_groups[0]['lr']
            print(f'Learning rate: {learning_rate:.8f} ')

            print(f'Epoch [{epoch + 1}/{num_epochs}], training day5 extractor Loss: {avg_loss:.4f}, '
                  f'Loss cl: {avg_loss_cl:.4f}, '
                  f'Loss supcon: {avg_loss_supcon:.4f}, '
                  f'Loss focal: {avg_loss_focal:.4f}, '
                  f'Loss bitemp: {avg_loss_bitemp:.4f}')

            writer.add_scalar('Loss/average_cl_loss', avg_loss_cl, epoch)
            writer.add_scalar('Loss/average_supcon_loss', avg_loss_supcon, epoch)
            writer.add_scalar('Loss/average_focal_loss', avg_loss_focal, epoch)
            writer.add_scalar('Loss/average_bitemp_loss', avg_loss_bitemp, epoch)
            writer.add_scalar('Loss/average_loss', avg_loss, epoch)

            del avg_loss_cl, avg_loss_supcon, avg_loss_focal, avg_loss_bitemp
            torch.cuda.empty_cache()

            accuracy = model_test(ema_efe_day5, ema_ecp, test_dataloader, criterion, epoch, writer, image_type='day5',
                                  label_type=label_type, test_type='Test day5')

        # 訓練day3_EFE與ECP subnetwork
        elif epoch >= step1 and epoch < step2:
            set_requires_grad(ECP, False)
            loss, loss_cl, loss_focal, loss_supcon, loss_bitemp = \
                trainer_cls_with_decoder(extractor=EFE, classifier=ECP, decoder=feature_decoder,
                                         classifier_optimizer=EFE_optimizer_with_decoder,
                                         dataloader=dataloader, ema_extor=ema_efe, ema_clsfer=ema_ecp,
                                         image_type='day3', device=device)

            avg_loss = (loss / batch_size)
            avg_loss_cl = (loss_cl / batch_size)
            avg_loss_focal = (loss_focal / batch_size)
            avg_loss_supcon = (loss_supcon / batch_size)
            avg_loss_bitemp = (loss_bitemp / batch_size)
            # scheduler_EFE.step()
            # scheduler_day3.step()

            learning_rate = EFE_optimizer_with_decoder.param_groups[0]['lr']
            print(f'Learning rate day3: {learning_rate:.8f} ')

            print(f'Epoch day3 [{epoch + 1}/{num_epochs}], training day3 extractor Loss: {avg_loss:.4f}, '
                  f'Loss cl: {avg_loss_cl:.4f}, '
                  f'Loss supcon: {avg_loss_supcon:.4f}, '
                  f'Loss focal: {avg_loss_focal:.4f}, '
                  f'Loss bitemp: {avg_loss_bitemp:.4f}')

            writer.add_scalar('Loss day3/average_cl_loss', avg_loss_cl, epoch)
            writer.add_scalar('Loss day3/average_supcon_loss', avg_loss_supcon, epoch)
            writer.add_scalar('Loss day3/average_focal_loss', avg_loss_focal, epoch)
            writer.add_scalar('Loss day3/average_bitemp_loss', avg_loss_bitemp, epoch)
            writer.add_scalar('Loss day3/average_loss', avg_loss, epoch)

            del avg_loss_cl, avg_loss_supcon, avg_loss_focal, avg_loss_bitemp
            torch.cuda.empty_cache()
            set_requires_grad(ECP, True)

            model_test_with_decoder(ema_efe, ema_ecp, feature_decoder, test_dataloader, epoch, writer,
                                    image_type=image_type, label_type=label_type, test_type='Test')

        # 訓練FM subnetwork
        elif epoch >= step2 and epoch < step3:
            total_loss, total_gp = trainer_2_extractor_dis(extractor=EFE, extractor2=EFE_day5,
                                                           discriminator=FM, decoder=feature_decoder,
                                                           discriminator_optimizer=FM_optimizer,
                                                           loss_fn=wgan_gp_loss,
                                                           dataloader=dataloader_gan, device=device)
            # scheduler_FM.step()

            avg_loss = (total_loss / batch_size)
            avg_gp = (total_gp / batch_size)
            gp_value = avg_gp

            del total_loss, total_gp
            torch.cuda.empty_cache()

            print(f'Step S2 Epoch [{epoch + 1}/{num_epochs}], FM Loss: {avg_loss:.4f}, gp: {avg_gp:.4f}')

        # 透過day5_EFE與FM subnetwork引導訓練day3_EFE subnetwork
        elif epoch >= step3:
            for i in range(loop_s):
                training_dis = True
                # 確保 gp_value 小於所設置閥值，確保FM subnetwork收斂穩定，以用於引導day3 EFE subnetwork
                while gp_value > threshold or training_dis is True:
                    training_dis = False
                    for j in range(s3):
                        set_requires_grad(EFE_day5, False)
                        set_requires_grad(EFE, False)
                        set_requires_grad(ECP, False)
                        total_loss, total_gp = trainer_2_extractor_dis(extractor=EFE, extractor2=EFE_day5,
                                                                       discriminator=FM, decoder=feature_decoder,
                                                                       discriminator_optimizer=FM_optimizer,
                                                                       loss_fn=wgan_gp_loss,
                                                                       dataloader=dataloader_gan, device=device)
                        # scheduler_FM.step()

                        avg_loss = (total_loss / batch_size)
                        avg_gp = (total_gp / batch_size)
                        gp_value = avg_gp
                        del total_loss, total_gp
                        torch.cuda.empty_cache()

                        print(
                            f'Step S2 Epoch [{epoch + 1}/{num_epochs}], S3[{j + 1}/{s3}], FM Loss: {avg_loss:.4f}, '
                            f'gp: {avg_gp:.4f}')

                        FM_subnetwork_model_save_path = os.path.join(path, 'FM_subnetwork_last.pth')
                        torch.save(FM.module.state_dict(), FM_subnetwork_model_save_path)

                        set_requires_grad(EFE_day5, True)
                        set_requires_grad(EFE, True)
                        set_requires_grad(ECP, True)

                if gp_value <= threshold:
                    # 通過收斂FM_subnetwork引導day3 EFE subnetwork
                    set_requires_grad(EFE, True)
                    set_requires_grad(EFE_day5, False)
                    set_requires_grad(FM, False)
                    set_requires_grad(ECP, False)

                    for te in range(1):
                        total_loss = trainer_extractor_with_decoder(extractor=EFE, extractor2=EFE_day5,
                                                                    discriminator=FM,
                                                                    decoder=feature_decoder,
                                                                    extr_optimizer=EFE_optimizer_with_decoder,
                                                                    dataloader=dataloader_gan, ema_extor=ema_efe,
                                                                    device=device)
                        # scheduler_EFE.step()
                        avg_loss = (total_loss / batch_size)
                        del total_loss
                        torch.cuda.empty_cache()
                        print(f'Epoch: [{epoch + 1}/{num_epochs}] Loops: [{i + 1}/{loop_s}], '
                              f'Train Extractor EFE loss:{avg_loss:.4f}')

                    s3 = 2 if avg_loss > 1.0 else (3 if avg_loss < -1.0 else 1)

                    set_requires_grad(EFE_day5, True)
                    set_requires_grad(FM, True)
                    set_requires_grad(ECP, True)

                    efe_iter += 1
                    accuracy = model_test_with_decoder(ema_efe, ema_ecp, feature_decoder, test_dataloader, efe_iter,
                                                       writer, image_type=image_type, label_type=label_type,
                                                       test_type='EFE training test Gen')

                    if accuracy >= temp_accuracy_d3:
                        temp_accuracy_d3 = accuracy

                        EFE_subnetwork_model_save_path = os.path.join(path, 'EFE_subnetwork_best.pth')
                        feature_decoder_model_save_path = os.path.join(path, 'feature_decoder_best.pth')
                        EMA_EFE_subnetwork_model_save_path = os.path.join(path, 'EMA_EFE_subnetwork_best.pth')
                        EMA_ECP_subnetwork_model_save_path = os.path.join(path, 'EMA_ECP_subnetwork_best.pth')

                        torch.save(EFE.module.state_dict(), EFE_subnetwork_model_save_path)
                        torch.save(feature_decoder.module.state_dict(), feature_decoder_model_save_path)
                        save_ema(ema_efe, EMA_EFE_subnetwork_model_save_path)
                        save_ema(ema_ecp, EMA_ECP_subnetwork_model_save_path)

                    for step in range(s5):
                        # 通過分類預測結果修正day3 EFE subnetwork
                        set_requires_grad(ECP, False)
                        loss_3, loss_cl_3, loss_focal_3, loss_supcon_3, loss_bitemp_3 = \
                            trainer_cls_with_decoder(extractor=EFE, classifier=ECP, decoder=feature_decoder,
                                                     classifier_optimizer=EFE_optimizer_with_decoder,
                                                     dataloader=dataloader, ema_extor=ema_efe, ema_clsfer=ema_ecp,
                                                     image_type='day3', device=device)
                        # scheduler_EFE.step()

                        learning_rate = EFE_optimizer_with_decoder.param_groups[0]['lr']
                        print(f'Learning rate: {learning_rate:.8f} ')

                        avg_loss = (loss_3 / batch_size)
                        avg_loss_cl = (loss_cl_3 / batch_size)
                        avg_loss_focal = (loss_focal_3 / batch_size)
                        avg_loss_supcon = (loss_supcon_3 / batch_size)
                        avg_loss_bitemp = (loss_bitemp_3 / batch_size)

                        print(f'Epoch [{epoch + 1}/{num_epochs}], S5[{step + 1}/{s5}], Loss: {avg_loss:.4f}, '
                              f'Loss cl: {avg_loss_cl:.4f}, '
                              f'Loss supcon: {avg_loss_supcon:.4f}, '
                              f'Loss focal: {avg_loss_focal:.4f}, '
                              f'Loss bitemp: {avg_loss_bitemp:.4f}')

                        del avg_loss_cl, avg_loss_supcon, avg_loss_focal, avg_loss_bitemp, avg_loss
                        del loss_3, loss_cl_3, loss_focal_3, loss_supcon_3, loss_bitemp_3
                        torch.cuda.empty_cache()

                        efe_iter += 1
                        accuracy = model_test_with_decoder(ema_efe, ema_ecp, feature_decoder, test_dataloader, efe_iter,
                                                           writer, image_type=image_type, label_type=label_type,
                                                           test_type='EFE training test cls')

                        if accuracy >= temp_accuracy_d3:
                            temp_accuracy_d3 = accuracy

                            EFE_subnetwork_model_save_path = os.path.join(path, 'EFE_subnetwork_best.pth')
                            feature_decoder_model_save_path = os.path.join(path, 'feature_decoder_best.pth')
                            EMA_EFE_subnetwork_model_save_path = os.path.join(path, 'EMA_EFE_subnetwork_best.pth')
                            EMA_ECP_subnetwork_model_save_path = os.path.join(path, 'EMA_ECP_subnetwork_best.pth')

                            torch.save(EFE.module.state_dict(), EFE_subnetwork_model_save_path)
                            torch.save(feature_decoder.module.state_dict(), feature_decoder_model_save_path)
                            save_ema(ema_efe, EMA_EFE_subnetwork_model_save_path)
                            save_ema(ema_ecp, EMA_ECP_subnetwork_model_save_path)

        if epoch % 1 == 0:
            if epoch >= 0 and epoch % 10 == 0:
                EFE_subnetwork_model_save_path = os.path.join(path, 'EFE_subnetwork_last.pth')
                EFE_day5_subnetwork_model_save_path = os.path.join(path, 'EFE_day5_subnetwork_last.pth')
                ECP_subnetwork_model_save_path = os.path.join(path, 'ECP_subnetwork_last.pth')
                FM_subnetwork_model_save_path = os.path.join(path, 'FM_subnetwork_last.pth')

                torch.save(EFE_day5.module.state_dict(), EFE_day5_subnetwork_model_save_path)
                torch.save(EFE.module.state_dict(), EFE_subnetwork_model_save_path)
                torch.save(ECP.module.state_dict(), ECP_subnetwork_model_save_path)
                torch.save(FM.module.state_dict(), FM_subnetwork_model_save_path)

                feature_decoder_model_save_path = os.path.join(path, 'feature_decoder_last.pth')
                torch.save(feature_decoder.module.state_dict(), feature_decoder_model_save_path)

            if accuracy > temp_accuracy:
                temp_accuracy = accuracy

                EFE_day5_subnetwork_model_save_path = os.path.join(path, 'EFE_day5_subnetwork_best.pth')
                ECP_subnetwork_model_save_path = os.path.join(path, 'ECP_subnetwork_best.pth')

                torch.save(EFE_day5.module.state_dict(), EFE_day5_subnetwork_model_save_path)
                torch.save(ECP.module.state_dict(), ECP_subnetwork_model_save_path)

            # torch.cuda.empty_cache()


if __name__ == '__main__':
    # 訓練結果與tensorboard紀錄結果儲存設置
    weight_save_file_path = "checkpoints_0617"
    os.makedirs(weight_save_file_path, exist_ok=True)
    day_3_to_5_path = os.path.join(weight_save_file_path, 'day3_to_day5_classification_log')

    writer = SummaryWriter(log_dir=day_3_to_5_path)
    train(image_type='day3', label_type='day5', writer=writer, path=day_3_to_5_path)
    writer.close()


