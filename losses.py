from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


# Supervised Contrastive loss function
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1.0, dtype=mask_pos_pairs.dtype).to(device),
                                     mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean', device='cpu'):
        """
        Initialize the FocalLoss module.

        :param alpha: Per-class scaling factors for the loss. Should be a tensor of shape [num_classes].
        :param gamma: Focusing parameter. Default is 2.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default is 'mean'.
        :param device: The device to run the computations on. Default is 'cpu'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha.to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for computing the focal loss.

        :param inputs: Predictions (logits) from the model (shape: [batch_size, num_classes]).
        :param targets: Ground truth labels (shape: [batch_size]).
        :return: Computed focal loss.
        """
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float().to(inputs.device)

        # Compute log softmax probabilities
        log_probs = F.log_softmax(inputs, dim=1)

        # Compute softmax probabilities
        probs = torch.exp(log_probs)

        # Compute the focal weight
        focal_weight = torch.pow(1 - probs, self.gamma)

        # Apply per-class alpha weight
        alpha_weight = self.alpha[targets].to(inputs.device).unsqueeze(1)

        # Compute the focal loss
        focal_loss = -alpha_weight * focal_weight * targets_one_hot * log_probs

        # Sum the losses over the classes
        focal_loss = focal_loss.sum(dim=1)

        # Apply reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BitemperedLoss(nn.Module):
    def __init__(self, num_classes, t1=.5, t2=1.5, reduction='mean'):
        super(BitemperedLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.num_classes = num_classes
        self.reduction = reduction
        self.cw = torch.tensor([2, 5, 1])

    @staticmethod
    def tempered_softmax(x, t, dim):
        """Compute the tempered softmax of x."""
        exp_x_t = torch.exp(x / t)
        return exp_x_t / torch.sum(exp_x_t, dim=dim, keepdim=True)

    @staticmethod
    def tempered_log(x, t):
        """Compute the tempered logarithm of x."""
        return (x ** (1 - t) - 1) / (1 - t)

    def forward(self, logits, labels):
        """
        Compute the bi-tempered logistic loss.
        logits: Predicted logits for each class.
        labels: True labels in one-hot encoded form.
        t1: Temperature parameter controlling the softness of the log function.
        t2: Temperature parameter controlling the confidence of the predictions.
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        """
        probs = self.tempered_softmax(logits, self.t2, dim=-1)
        log_probs = self.tempered_log(probs, self.t1)
        loss = -torch.sum(labels * log_probs, dim=-1)
        cw = torch.argmax(labels, 1)
        loss = loss * self.cw.to(labels.device)[cw].unsqueeze(0)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss



def wasserstein_loss(real_scores, fake_scores):
    # with torch.cuda.amp.autocast(True):
    return torch.mean(fake_scores) - torch.mean(real_scores)

# Combined WGAN-GP loss function
def wgan_gp_loss(critic, real_data, fake_data, lambda_gp=30, gamma=1, device='cuda'):
    # Calculate Wasserstein loss
    real_scores = critic(real_data)
    fake_scores = critic(fake_data)
    wasserstein_loss_value = torch.mean(fake_scores) - torch.mean(real_scores)

    # Calculate gradient penalty
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    interpolated_data = alpha * real_data + (1 - alpha) * fake_data
    interpolated_data.requires_grad_(True)
    interpolated_scores = critic(interpolated_data)
    gradients = torch.autograd.grad(outputs=interpolated_scores, inputs=interpolated_data,
                                    grad_outputs=torch.ones(interpolated_scores.size()).to(device),
                                    create_graph=True, retain_graph=True)[0]
    gradient_penalty = (((gradients.norm(2, dim=1) - gamma) / gamma) ** 2).mean()

    # Combine Wasserstein loss and gradient penalty
    wgan_gp_loss = wasserstein_loss_value + lambda_gp * gradient_penalty
    gradient_penalty = gradient_penalty.item()
    torch.cuda.empty_cache()

    return wgan_gp_loss, gradient_penalty