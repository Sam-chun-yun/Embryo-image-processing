import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from losses import SupConLoss, FocalLoss, BitemperedLoss
import torch.nn.functional as F

# AMP初始化
scaler = GradScaler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義損失函數
weights = torch.tensor([2.5, 1.0], dtype=torch.float16).to(device)
alpha = torch.tensor([2.5, 1.0], dtype=torch.float16).to(device)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.03, weight=weights)  # 使用標準交叉熵損失

# 初始化 Focal Loss
temperature = 0.07
SupConLoss = SupConLoss(temperature)
focal_loss_criterion = FocalLoss(alpha, gamma=2, reduction='mean', device=device)
bitemp_loss_criterion = BitemperedLoss(num_classes=2)


def trainer_cls(extractor, classifier, classifier_optimizer, dataloader, ema_extor, ema_clsfer, image_type, device):
    total_loss = 0.0
    total_loss_cl = 0.0
    total_loss_supcon = 0.0
    total_loss_focal = 0.0
    total_loss_bitemp = 0.0
    classifier_optimizer.zero_grad()

    for batch_idx, (day3_img, day5_img, label, _) in enumerate(tqdm(dataloader, desc="Processing batches", position=0)):
        extractor.train()
        classifier.train()

        if image_type == 'day3':
            input_images = day3_img
        elif image_type == 'day5':
            input_images = day5_img
        input_label = label.to(device)

        # 前向傳播
        with autocast():
            output_x4, output_x2, output = extractor(input_images)
            pred, logits, logits_2 = classifier(output_x4, output_x2, output)

        features = F.normalize(logits, dim=1).unsqueeze(1)
        # features_2 = F.normalize(logits_2, dim=1).unsqueeze(1)
        # features = torch.cat((features_1, features_2), dim=1)

        one_hot_label = torch.eye(3)[input_label].to(device)

        loss_supcon = SupConLoss(features, input_label) * 0.2
        # loss_supcon = 0
        loss_focal = focal_loss_criterion(pred, input_label) * 2

        loss_cls = criterion(pred, input_label)
        # loss_cls =0
        # loss_bitemp = bitemp_loss_criterion(pred, one_hot_label)
        loss_bitemp = 0
        loss = loss_cls + loss_supcon + loss_focal + loss_bitemp

        scaler.scale(loss).backward()
        scaler.step(classifier_optimizer)
        scaler.update()
        classifier_optimizer.zero_grad()

        ema_extor.update()
        ema_clsfer.update()

        total_loss_cl += loss_cls
        total_loss_focal += loss_focal
        total_loss_supcon += loss_supcon
        total_loss_bitemp += loss_bitemp
        total_loss += loss.item()

    return total_loss, total_loss_cl, total_loss_focal, total_loss_supcon, total_loss_bitemp


def trainer_discriminator(extractor, discriminator, discriminator_optimizer, loss_fn, dataloader, device='cuda'):
    total_gp = 0
    total_loss = 0
    discriminator.train()
    discriminator_optimizer.zero_grad()

    for batch_idx, (day3_img, day5_img, _, _) in enumerate(tqdm(dataloader, desc="Processing batches", position=0)):
        with autocast():
            input_images_feature, _, _ = extractor(day3_img)
            input_images_target_feature, _, _ = extractor(day5_img)
        loss, gp = loss_fn(discriminator, input_images_target_feature, input_images_feature, device=device)

        scaler.scale(loss).backward()
        scaler.step(discriminator_optimizer)
        scaler.update()
        discriminator_optimizer.zero_grad()
        total_gp += gp
        total_loss += loss.item()

    return total_loss, total_gp


def trainer_extractor(extractor, discriminator, extr_optimizer, dataloader, ema_extor):
    total_loss = 0.0
    extractor.train()
    extr_optimizer.zero_grad()

    for batch_idx, (day3_img, _, _) in enumerate(tqdm(dataloader, desc="Processing batches", position=0)):
        with autocast():
            input_feature, _, _ = extractor(day3_img)
            loss = -1 * (discriminator(input_feature).mean())
            total_loss += loss

        scaler.scale(loss).backward()
        scaler.step(extr_optimizer)
        scaler.update()
        ema_extor.update()

        extr_optimizer.zero_grad()

    return total_loss


def trainer_extractor_with_decoder(extractor, extractor2, discriminator, decoder, extr_optimizer, dataloader, ema_extor,
                                   device):
    total_loss = 0.0
    smooth_L1 = torch.nn.SmoothL1Loss()

    extractor.train()
    decoder.train()
    extr_optimizer.zero_grad()

    for batch_idx, (day3_img, day5_img, _, _) in enumerate(tqdm(dataloader, desc="Processing batches", position=0)):
        with autocast():
            input_feature, input_feature_2, input_feature_3 = extractor(day3_img)

            input_feature_day5, input_feature_day5_2, input_feature_day5_3 = extractor2(day5_img)

            decoder_image = decoder(input_feature_3)

            day3_img = torch.add(day3_img.to(device), decoder_image)

            input_feature, input_feature_2, input_feature_3 = extractor(day3_img)

            image_domain_loss = smooth_L1(decoder_image, day5_img.to(device))
            image_domain_loss += smooth_L1(input_feature, input_feature_day5) + \
                                 smooth_L1(input_feature_2, input_feature_day5_2) + \
                                 smooth_L1(input_feature_3, input_feature_day5_3)

            loss = -1 * (discriminator(input_feature).mean()) + image_domain_loss

            # loss = image_domain_loss

        scaler.scale(loss).backward()
        scaler.step(extr_optimizer)
        scaler.update()
        ema_extor.update()
        total_loss += loss.item()

        extr_optimizer.zero_grad()

    return total_loss


def trainer_2_extractor_dis(extractor, extractor2, discriminator, decoder, discriminator_optimizer, loss_fn, dataloader,
                            device='cuda'):
    total_gp = 0
    total_loss = 0
    discriminator.train()
    discriminator_optimizer.zero_grad()

    for batch_idx, (day3_img, day5_img, _, _) in enumerate(tqdm(dataloader, desc="Processing batches", position=0)):
        with autocast():
            input_images_feature, _, images_feature = extractor(day3_img)
            sny_img = decoder(images_feature)
            day3_img = torch.add(day3_img.to(device), sny_img)

            input_images_feature, _, _ = extractor(day3_img)

            input_images_target_feature, _, _ = extractor2(day5_img)

        loss, gp = loss_fn(discriminator, input_images_target_feature, input_images_feature, device=device)

        scaler.scale(loss).backward()
        scaler.step(discriminator_optimizer)
        scaler.update()
        discriminator_optimizer.zero_grad()
        total_gp += gp
        total_loss += loss.item()

    return total_loss, total_gp


def trainer_cls_with_decoder(extractor, classifier, decoder, classifier_optimizer, dataloader, ema_extor, ema_clsfer,
                             image_type, device):
    total_loss = 0.0
    total_loss_cl = 0.0
    total_loss_supcon = 0.0
    total_loss_focal = 0.0
    total_loss_bitemp = 0.0
    classifier_optimizer.zero_grad()
    smoothL1_loss = torch.nn.SmoothL1Loss()

    for batch_idx, (day3_img, day5_img, label, _) in enumerate(tqdm(dataloader, desc="Processing batches", position=0)):
        extractor.train()
        classifier.train()

        if image_type == 'day3':
            input_images = day3_img
        elif image_type == 'day5':
            input_images = day5_img
        input_label = label.to(device)

        # 前向傳播
        with autocast():
            output_x4, output_x2, output = extractor(input_images)
            decoder_image = decoder(output)
            input_images = torch.add(input_images.to(device), decoder_image)
            output_x4, output_x2, output = extractor(input_images)

            pred, logits, logits_2 = classifier(output_x4, output_x2, output)

        features = F.normalize(logits, dim=1).unsqueeze(1)
        # features_2 = F.normalize(logits_2, dim=1).unsqueeze(1)
        # features = torch.cat((features_1, features_2), dim=1)

        # one_hot_label = torch.eye(3)[input_label].to(device)

        loss_supcon = SupConLoss(features, input_label) * 0.2
        # loss_supcon = 0
        loss_focal = focal_loss_criterion(pred, input_label) * 2
        # loss_focal = 0
        loss_cls = criterion(pred, input_label)
        # loss_bitemp = bitemp_loss_criterion(pred, one_hot_label)
        loss_bitemp = 0
        loss_smoothL1 = smoothL1_loss(decoder_image, day5_img.to(device))
        # loss_smoothL1 = 0

        # loss_bitemp = 0
        loss = loss_cls + loss_supcon + loss_focal + loss_bitemp + loss_smoothL1

        scaler.scale(loss).backward()
        scaler.step(classifier_optimizer)
        scaler.update()
        classifier_optimizer.zero_grad()

        ema_extor.update()
        ema_clsfer.update()

        total_loss_cl += loss_cls
        total_loss_focal += loss_focal
        total_loss_supcon += loss_supcon
        total_loss_bitemp += loss_bitemp
        total_loss += loss.item()

    return total_loss, total_loss_cl, total_loss_focal, total_loss_supcon, total_loss_bitemp