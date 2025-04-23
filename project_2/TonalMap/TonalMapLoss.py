import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from cv2.gapi.wip import GOutputs


class VGGFeatures(nn.Module):
    def __init__(self, layer=15, requires_grad=False):
        super(VGGFeatures, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features
        if not requires_grad:
            for param in self.vgg19.parameters():  # Corrected line
                param.requires_grad = False

    def forward(self, y):
        features = self.vgg19(y)
        return features

class VGG19FeatureExtractor(nn.Module):

    def __init__(self, feature_type='VGG54', requires_grad=False):
        super().__init__()

        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features

        # Store blocks like conv1, conv2, etc.
        self.blocks = nn.ModuleDict({
            'conv1': nn.Sequential(*vgg[:4]),  # conv1_1, conv1_2
            'pool1': vgg[4:5],
            'conv2': nn.Sequential(*vgg[5:9]),  # conv2_1, conv2_2
            'pool2': vgg[9:10],
            'conv3': nn.Sequential(*vgg[10:16]),  # conv3_1 - conv3_4
            'pool3': vgg[16:17],
            'conv4': nn.Sequential(*vgg[17:23]),  # conv4_1 - conv4_4
            'pool4': vgg[23:24],
            'conv5': nn.Sequential(*vgg[24:30]),  # conv5_1 - conv5_4
            'pool5': vgg[30:31],
        })

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        # Map Slim-style names to block/layer index
        self.feature_map = {
            'VGG11': ('conv1', 0),
            'VGG21': ('conv2', 0),
            'VGG22': ('conv2', 1),
            'VGG31': ('conv3', 0),
            'VGG34': ('conv3', 3),
            'VGG41': ('conv4', 0),
            'VGG51': ('conv5', 0),
            'VGG54': ('conv5', 3),
        }

        if feature_type not in self.feature_map:
            raise ValueError(f"Unknown feature type: {feature_type}")

        self.target_block, self.target_index = self.feature_map[feature_type]

    def forward(self, x):
        for name, layer in self.blocks.items():
            x = layer(x)
            if name == self.target_block:
                # convX block is a Sequential with multiple conv layers (and relus)
                return x[:, :, :, :] if self.target_index is None else x

        raise RuntimeError(f"Feature layer {self.target_block} not reached in forward pass.")

class FeatureContrastMasking(nn.Module):
    def __init__(self, gamma = 0.5, beta = 0.5, kernel_size=13, sigma=2.0, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gamma = gamma
        self.beta = beta

    def local_mean_std(self, input_feature):
        B, C, H, W = input_feature.shape
        device = input_feature.device

        # Gaussian
        gaussian_blur = T.GaussianBlur(self.kernel_size, self.sigma)
        mean_local_gauss = gaussian_blur(input_feature)
        # Box
        kernel = torch.ones((1, 1, self.kernel_size, self.kernel_size), dtype=torch.float32, device=device) / self.kernel_size ** 2
        kernel = kernel.repeat(C, 1, 1, 1)
        mean_local_box = F.conv2d(input_feature, kernel, padding=self.kernel_size // 2, groups=C)
        square_local_box = F.conv2d(torch.pow(input_feature, 2), kernel, padding=self.kernel_size // 2, groups=C)
        square_mean_local_box = torch.clip(torch.pow(mean_local_box, 2), 10e-8)
        diff = square_local_box - square_mean_local_box
        std_local = torch.sqrt(torch.abs(diff) + 10e-8)

        return mean_local_gauss, std_local, mean_local_box

    def forward(self, input_feature):
        """
        Args:
            input_feature (Tensor): shape [B, C, H, W]

        Returns:
            contrast_map (Tensor): same shape as input
        """
        mean_local_gauss, std_local, mean_local_box = self.local_mean_std(input_feature)
        # print(torch.max(mean_local_gauss))
        # print(torch.min(mean_local_gauss))
        # print(torch.max(std_local))
        # print(torch.min(std_local))
        # print(torch.max(mean_local_box))
        # print(torch.min(mean_local_box))

        gaussian_norm = (input_feature - mean_local_gauss) / (torch.abs(mean_local_gauss) + 10e-8)

        msk = torch.where(torch.greater(gaussian_norm, 0.0), torch.ones_like(gaussian_norm),
                          (-1) * torch.zeros_like(gaussian_norm))
        gaussian_norm = torch.where(torch.eq(gaussian_norm, 0.0), 10e-8 * torch.ones_like(gaussian_norm), gaussian_norm)
        # print(torch.max(gaussian_norm))
        # print(torch.min(gaussian_norm))
        gaussian_norm = torch.pow(torch.abs(gaussian_norm), self.gamma)
        # print(torch.max(gaussian_norm))
        # print(torch.min(gaussian_norm))
        norm_num = msk * gaussian_norm

        # print(torch.max(msk))
        # print(torch.min(msk))


        local_norm = std_local / (torch.abs(mean_local_box) + 10e-8)
        norm_den = torch.pow(local_norm, self.beta)
        norm_den = 1.0 + norm_den

        return norm_num / norm_den

class AdaptiveMuLawCompression(torch.nn.Module):
    def __init__(self,
                 lambda1=8.759, gamma1=2.148,
                 lambda2=0.1494, gamma2=-2.067,
                 epsilon=1e-6):
        super().__init__()
        self.lambda1 = lambda1
        self.gamma1 = gamma1
        self.lambda2 = lambda2
        self.gamma2 = gamma2
        self.epsilon = epsilon

    def compute_mu(self, hdr_image):
        """
        Compute adaptive μ from median intensity of the HDR image.
        Assumes input is [B, C, H, W] with values in range [0, 1] or HDR range.
        """
        # Convert to grayscale luminance: average over channels
        B = hdr_image.shape[0]
        i_HDR = hdr_image.view(B, -1).median(dim=1).values
        #luminance = hdr_image.mean(dim=1, keepdim=True)  # shape: [B, 1, H, W]
        #B = luminance.shape[0]

        # Flatten and compute median per image in batch
        #i_HDR = torch.median(luminance.view(B, -1), dim=1).values  # [B]

        # Compute μ using fitted function (Eq. 3)
        mu = (
            self.lambda1 * (i_HDR ** self.gamma1) +
            self.lambda2 * (i_HDR ** self.gamma2)
        )

        return mu  # shape: [B]

    def forward(self, hdr_image):
        """
        Apply μ-law compression to HDR image.
        Args:
            hdr_image: torch.Tensor, shape [B, C, H, W]
        Returns:
            I_mu: torch.Tensor, shape [B, C, H, W]
        """
        #B, C, H, W = hdr_image.shape
        mu = self.compute_mu(hdr_image)  # [B]

        # Expand μ for broadcasting: [B, 1, 1, 1]
        #mu_expand = mu.view(B, 1, 1, 1)

        # μ-law compression (Eq. 2)
        #numerator = torch.log(1 + mu_expand * hdr_image)
        #denominator = torch.log(1 + mu_expand + self.epsilon)
        #I_mu = numerator / (denominator + self.epsilon)

        I_mu = torch.log1p(mu * hdr_image) / torch.log(torch.tensor(1.0 + mu))

        return I_mu


class TonalMapLoss(nn.Module):
    def __init__(self, vgg_layer=8):
        super(TonalMapLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.extract_features = VGG19FeatureExtractor('VGG21')
        self.calculate_masking = FeatureContrastMasking()
        self.adaptive_mu_compression = AdaptiveMuLawCompression()

    def forward(self, output, input, device):

        ## print(output)
        ## print(output.shape)
        ## print(input.shape)
        input_comp = self.adaptive_mu_compression(input)
        # print(torch.max(input_comp))
        # print(torch.min(input_comp))
        ## print(input_comp.shape)
        # print(torch.max(output))
        # print(torch.min(output))

        output_features = self.extract_features(output)
        input_features = self.extract_features(input_comp)

        # print(torch.max(output_features))
        # print(torch.min(output_features))
        # print(torch.max(input_features))
        # print(torch.min(input_features))

        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$calculating output fVGG")
        output_fVGG = self.calculate_masking(output_features)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$calculating input fVGG")
        input_fVGG = self.calculate_masking(input_features)

        # print(torch.max(output_fVGG))
        # print(torch.min(output_fVGG))
        # print(torch.max(input_fVGG))
        # print(torch.min(input_fVGG))

        ## print(output_fVGG)
        ## print(input_features)

        loss = self.l1_loss(output_fVGG, input_fVGG)
        ## print(loss)
        return loss