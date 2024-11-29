import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.cuda.amp import autocast

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = torch.cuda.amp.GradScaler()


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], load_weight=False):
    gpu_ids = [int(digit) for digit in list(gpu_ids)]
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    if load_weight is False:
        init_weights(net, init_type, gain=init_gain)
    return net


# set requies_grad=Fasle to avoid computation

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        with torch.cuda.amp.autocast():
            return torch.cat(x, self.d)


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x1 = self.cv4(self.cv3(self.cv1(x)))
            y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
            y2 = self.cv2(x)
            return self.cv7(torch.cat((y1, y2), dim=1))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        with torch.cuda.amp.autocast():
            m_batchsize, C, width, height = x.size()
            proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
            proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*  b H)
            energy = torch.bmm(proj_query, proj_key)  # transpose check
            attention = self.softmax(energy).to(torch.float16)  # BX (N) X (N)
            proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(m_batchsize, C, width, height)
            self.gamma = self.gamma.to(x.device)
            out = (self.gamma * out + x).to(torch.float16)
            return out


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            return self.m(x)


class AP_S(nn.Module):
    def __init__(self, k=2):
        super(AP_S, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            return self.m(x)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Conv_na(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_na, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.GRN = GRN(c2)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            # return self.GRN(self.act(self.bn(self.conv(x))))
            return self.bn(self.conv(x))


class Conv_GRN(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_GRN, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.GRN = GRN(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        # with torch.cuda.amp.autocast():
        # return self.act(self.bn(self.conv(x)))
        return self.GRN(self.act(self.bn(self.conv(x))))
            # return self.act(self.bn(self.conv(x)))


class Maxpool_Conv_neck(nn.Module):
    def __init__(self, c1):
        super(Maxpool_Conv_neck, self).__init__()
        c2 = int(c1 // 2)
        self.maxpool_conv = nn.Sequential(MP(),
                                          Conv(c1=c1, c2=c2, k=1, s=1))
        self.conv_conv = nn.Sequential(Conv(c1=c1, c2=c2, k=1, s=1),
                                       Conv(c1=c2, c2=c2, k=3, s=2))
        self.cat = Concat()

    def forward(self, x):
        # with torch.cuda.amp.autocast():
        path1_max = self.maxpool_conv(x)
        path2_conv = self.conv_conv(x)
        out = self.cat([path1_max, path2_conv])
        return out


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        # Removing the device specification here, it will inherit the device from the input tensor
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # with torch.cuda.amp.autocast():
        x = x.permute(0, 2, 3, 1)
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)

        gamma = self.gamma
        beta = self.beta

        x = gamma * (x * Nx) + beta + x
        x = x.permute(0, 3, 1, 2)
        return x


class ELAB_H(nn.Module):
    def __init__(self, c1, c2):
        super(ELAB_H, self).__init__()
        self.conv1 = Conv(c1=c1, c2=c1 // 2, k=1, s=1)
        self.conv2 = Conv(c1=c1, c2=c1 // 2, k=1, s=1)
        self.conv3 = Conv(c1=c1 // 2, c2=c1 // 4, k=3, s=1)
        self.conv4 = Conv(c1=c1 // 4, c2=c1 // 4, k=3, s=1)

        self.conv5 = Conv(c1=c1 // 4, c2=c1 // 4, k=1, s=1)
        self.conv6 = Conv(c1=c1 // 4, c2=c1 // 4, k=1, s=1)

        self.cat = Concat()
        # self.conv = Conv(c1=c1 * 2, c2=c2, k=1, s=1)
        self.conv = Conv_GRN(c1=c1 * 2, c2=c2, k=1, s=1)
        # self.GRN = GRN(c2)

    def forward(self, x):
        # with torch.cuda.amp.autocast():
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv4(p4)
        p6 = self.conv4(p5)

        out = self.cat([p1, p2, p3, p4, p5, p6])
        out = self.conv(out)
        # out = self.GRN(out)

        return out


class ELAB(nn.Module):
    def __init__(self, c1, c2):
        super(ELAB, self).__init__()
        self.conv1 = Conv(c1=c1, c2=c1 // 2, k=1, s=1)
        self.conv2 = Conv(c1=c1, c2=c1 // 2, k=1, s=1)
        self.conv3 = nn.Sequential(Conv(c1=c1 // 2, c2=c1 // 2, k=3, s=1),
                                   # Conv(c1=c1 // 2, c2=c1 // 2, k=3, s=1),
                                   Conv(c1=c1 // 2, c2=c1 // 2, k=3, s=1))
        self.conv4 = nn.Sequential(Conv(c1=c1 // 2, c2=c1 // 2, k=3, s=1),
                                   # Conv(c1=c1 // 2, c2=c1 // 2, k=3, s=1),
                                   Conv(c1=c1 // 2, c2=c1 // 2, k=3, s=1))

        self.cat = Concat()
        # self.conv = Conv(c1=c1 * 2, c2=c2, k=1, s=1)
        self.conv = nn.Sequential(Conv(c1=c1 * 2, c2=c2, k=3, s=1),
                                  Conv_GRN(c1=c2, c2=c2, k=1, s=1)
                                 )
        # self.GRN = GRN(c2)

    def forward(self, x):
        # with torch.cuda.amp.autocast():
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)

        out = self.cat([p1, p2, p3, p4])
        out = self.conv(out)
        # out = self.GRN(out)

        return out


class EFE_subnetwork(nn.Module):
    def __init__(self, c1=32):
        super(EFE_subnetwork, self).__init__()
        self.FEM = nn.Sequential(Conv(c1=3, c2=c1, k=3, s=1),
                                 Conv(c1=c1, c2=c1 * 2, k=3, s=2),
                                 Conv(c1=c1 * 2, c2=c1 * 2, k=3, s=1),
                                 Conv(c1=c1 * 2, c2=c1 * 4, k=3, s=2),
                                 ELAB(c1 * 4, c1 * 8),
                                 # ELAB(c1 * 8, c1 * 8),
                                 )

        self.FAM_2 = nn.Sequential(Maxpool_Conv_neck(c1 * 8),
                                   ELAB(c1 * 8, c1 * 16)
                                   # ELAB(c1 * 16, c1 * 16)
                                   )

        self.FAM_3 = nn.Sequential(Maxpool_Conv_neck(c1 * 16),
                                   ELAB(c1 * 16, c1 * 32),
                                   # ELAB(c1 * 32, c1 * 32),
                                   ELAB(c1 * 32, c1 * 32)
                                   )

    def forward(self, x):
        with torch.cuda.amp.autocast():
            # x = self.FEM(x)
            out_x4 = self.FEM(x)
            out_x2 = self.FAM_2(out_x4)
            out = self.FAM_3(out_x2)
            return out_x4, out_x2, out

class Conv_LeakyReLU(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_LeakyReLU, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        with torch.cuda.amp.autocast():
            return self.act(self.bn(self.conv(x)))

class FM_subnetwork(nn.Module):
    def __init__(self, c1=256):
        super(FM_subnetwork, self).__init__()
        self.Attention = nn.Sequential(nn.MaxPool2d(2, stride=2),
                                       Self_Attn(c1))

        self.Dis = nn.Sequential(Conv_LeakyReLU(c1=c1,     c2=c1,     k=5, s=2),
                                 Conv_LeakyReLU(c1=c1,     c2=c1 * 2, k=5, s=2),
                                 Conv_LeakyReLU(c1=c1 * 2, c2=c1 * 4, k=5, s=2),
                                 Conv_LeakyReLU(c1=c1 * 4, c2=c1 * 4, k=5, s=2),
                                 nn.Conv2d(c1 * 4, 1, 3, 1),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x = self.Attention(x)
            x = self.Dis(x)
            return x

class Decoder(nn.Module):
    def __init__(self, c1=1024):
        super(Decoder, self).__init__()
        # c2 = int(c1 / 2)
        self.decoder = nn.Sequential(Conv(c1=c1, c2=512, k=1, s=1),
                                     nn.Upsample(scale_factor=2, mode='nearest'),  # 256, 160, 160
                                     Conv(c1=512, c2=512, k=3, s=1),
                                     Conv(c1=512, c2=256, k=1, s=1),
                                     nn.Upsample(scale_factor=2, mode='nearest'),  # 256, 160, 160
                                     Conv(c1=256, c2=256, k=3, s=1),
                                     Conv(c1=256, c2=128, k=1, s=1),
                                     nn.Upsample(scale_factor=2, mode='nearest'),  # 256, 80, 80
                                     Conv(c1=128, c2=128, k=3, s=1),
                                     Conv(c1=128, c2=64, k=1, s=1),
                                     nn.Upsample(scale_factor=2, mode='nearest'),
                                     Conv(c1=64, c2=64, k=3, s=1),
                                     Conv(c1=64, c2=3, k=3, s=1),
                                     nn.Tanh(),
                                     )

    def forward(self, x):
        x = self.decoder(x)
        return x



# Embryo Classification Prediction (ECP) Subnetwork
class ECP_subnetwork(nn.Module):
    def __init__(self, c1=1024, num_class=3):
        super(ECP_subnetwork, self).__init__()
        c2 = int(c1 / 2)
        c4 = int(c1 / 4)
        c = int(c1 * 2)
        # Feature Fusion Module
        self.avgpool_x4_conv = nn.Sequential(AP_S(4),
                                             Conv(c1=c4, c2=c4, k=3, s=1),
                                             Conv(c1=c4, c2=c4, k=1, s=1))

        self.avgpool_x2_conv = nn.Sequential(AP_S(2),
                                             Conv(c1=c2, c2=c2, k=3, s=1),
                                             Conv(c1=c2, c2=c4, k=1, s=1))

        self.conv1 = Conv(c1=c2, c2=c2, k=3, s=1)
        self.conv2 = Conv(c1=c1, c2=c1, k=1, s=1)
        self.conv3 = Conv(c1=c1 + c2, c2=c1, k=3, s=1)
        self.cat = Concat()
        self.SPPCSPC = nn.Sequential(SPPCSPC(c1, c1),
                                     nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                     nn.Flatten())

        # Classification Module
        self.EC_module = nn.Sequential(nn.Linear(c1, c1),
                                       nn.SiLU(),
                                       nn.Linear(c1, num_class)
        )

    def forward(self, x_x4, x_x2, x):
        with torch.cuda.amp.autocast():
            out = self.cat([self.avgpool_x4_conv(x_x4), self.avgpool_x2_conv(x_x2)])
            out = self.conv3(self.cat([self.conv1(out), self.conv2(x)]))
            out = self.SPPCSPC(out)
            out = self.EC_module(out)
            return out


class ECP_subnetwork_logit(nn.Module):
    def __init__(self, c1=1024, num_class=2):
        super(ECP_subnetwork_logit, self).__init__()
        c2 = int(c1 / 2)
        c4 = int(c1 / 4)
        c = int(c1 * 2)
        # Feature Fusion Module
        self.avgpool_x4_conv = nn.Sequential(
            AP_S(2),
            Conv(c1=c4, c2=c4, k=3, s=1),
            Conv_GRN(c1=c4, c2=c4, k=1, s=1),
            AP_S(2),
            Conv(c1=c4, c2=c4, k=3, s=1),
            Conv_GRN(c1=c4, c2=c4, k=1, s=1))

        self.avgpool_x2_conv = nn.Sequential(
            AP_S(2),
            Conv(c1=c2, c2=c2, k=3, s=1),
            Conv(c1=c2, c2=c4, k=1, s=1))

        self.conv1 = nn.Sequential(Conv(c1=c1 + c4, c2=c1 + c4, k=3, s=1),
                                   Conv_GRN(c1=c1 + c4, c2=c1, k=1, s=1)
                                   )
        self.conv2 = Conv(c1=c1, c2=c1, k=1, s=1)

        self.conv3 = nn.Sequential(Conv(c1=c1 + c4, c2=c1 + c4, k=3, s=1),
                                   Conv_GRN(c1=c1 + c4, c2=c1 + c4, k=1, s=1)
                                   )
        c = c1
        c1 = c1 + c4
        self.cat = Concat()
        self.SPPCSPC = nn.Sequential(
            SPPCSPC(c1, c1),
            Maxpool_Conv_neck(c1),
            ELAB_H(c1=c1, c2=c1),
            # Maxpool_Conv_neck(c1),
            # ELAB_H(c1=c1, c2=c1),
            # MP(),
            Conv(c1=c1, c2=c1, k=3, s=1, p=0),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten())

        # Classification Module
        self.EC_module = nn.Sequential(
            nn.Linear(c1, c1),
            nn.SiLU(),
            nn.Linear(c1, num_class)
        )

        self.project_head = nn.Sequential(nn.Linear(c1, 128),
                                          nn.SiLU(),
                                          nn.Linear(128, c1)
                                         )

        self.project_head_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(c, 128),
            nn.SiLU(),
            nn.Linear(128, c1)
        )

    def forward(self, x_x4, x_x2, x):
        with torch.cuda.amp.autocast():
            out = self.cat([self.conv2(x), self.avgpool_x2_conv(x_x2)])
            out = self.conv3(self.cat([self.conv1(out), self.avgpool_x4_conv(x_x4)]))
            logits = self.SPPCSPC(out)

            out = self.EC_module(logits)

            logits = self.project_head(logits)

            logits_2 = self.project_head_2(x)

            # if self.training:
            #     return out, logits, logits_2
            # else:
            #     return out

            return out, logits, logits_2

# model = EFE_subnetwork()
# model_2 = ECP_subnetwork()
# dis = FM_subnetwork()
#
# x = torch.randn(16, 3, 320, 320)
# out_x4, out_x2, out = model(x)
# c = model_2(out_x4, out_x2, out)
# p = dis(out)


