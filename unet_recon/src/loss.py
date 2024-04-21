import torch
import torch.nn as nn
from torchvision import models
from pdb import set_trace
from skimage.metrics import structural_similarity as ssim

class ReconLoss(nn.Module):
    """loss **to be used in the analysis** (not fine-tuning, to quantify 
    predictabilities 
    
    in:
    -extractor:
    -'alex' or 'vgg' or 'vgg-b' or 'vgg-bp' 
    
    todo: 
    - add eval_mask -- can we then still focus on the hole like we do now?
    """
    def __init__(self, extractor:str='alex',loss:str='L1', tv_loss='mean',
                 add_loss_suff:bool=False):
        super().__init__() 
        self.loss_str=loss
        self.loss = getattr(nn,f'{loss}Loss')(reduction='none')
        self.add_loss_suff=add_loss_suff
        if extractor=='alex':
            self.extractor = AlexNetFeatureExtractor()
        elif extractor=='vgg': # features from all layers
            self.extractor = VGGFullFeatureExtractor()
        elif extractor=='vgg-b':
            self.extractor=VGGBlockFeatureExtractor()
        elif extractor=='vgg-bp':
            self.extractor=VGGBlockFeatureExtractor(only_maxpool=True)
        else:
            raise ValueError('expecting "vgg" or "alex"')

    def forward(self, input, mask, output, gt):
        """todo: IF L1 vs L2 matters more than trivially, 
        include content_loss_str and include it in the name
        eg.
        content_loss-L1_0, content_loss-L1_1, etc 
        """
        # Non-hole pixels directly set to ground truth
        comp = mask * input + (1 - mask) * output

        # Hole Pixel Loss
        pixel_loss = (self.loss(comp,gt)).mean(dim=(1,2,3)).numpy()
        pixel_loss_hole = (self.loss((1-mask) * output, (1-mask) * gt)).mean(dim=(1,2,3)).numpy()

        # Perceptual Loss and Style Loss
        feats_comp = self.extractor(comp)
        feats_gt = self.extractor(gt)
        n_layers=len(feats_comp) # n_layers 

        # make dict for layered_losses 
        layered_losses=([f'content_loss_{li}' for li in range(n_layers)]+
                        [f'style_loss_{li}' for li in range(n_layers)])
        loss_dict={k:[] for k in ['pixel_loss']+layered_losses}

        # Calculate the L1Loss for each feature map
        loss_dict['pixel_loss']=pixel_loss
        loss_dict['pixel_loss_hole']=pixel_loss_hole

        for layer_i in range(n_layers ):
            loss_dict[f'content_loss_{layer_i}']=(
                self.loss(feats_comp[layer_i], feats_gt[layer_i]).mean(dim=(1,2,3)).numpy()) 
            loss_dict[f'style_loss_{layer_i}']=(
                self.loss(gram_matrix(feats_comp[layer_i]),
                          gram_matrix(feats_gt[layer_i])).mean(dim=(1,2)).numpy())
        if self.add_loss_suff:
            loss_dict = {key+f'_{self.loss_str}': value for key, value in loss_dict.items()}

        return(loss_dict)


class InpaintingLoss(nn.Module):
    """the actual loss to be used during training (calculates grand avg score)"""
    def __init__(self, extractor, tv_loss='mean'):
        super(InpaintingLoss, self).__init__()
        self.tv_loss = tv_loss
        self.l1 = nn.L1Loss()
        # default extractor is VGG16
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        # Non-hole pixels directly set to ground truth
        comp = mask * input + (1 - mask) * output

        # Total Variation Regularization
        tv_loss = total_variation_loss(comp, mask, self.tv_loss)
        # tv_loss = (torch.mean(torch.abs(comp[:, :, :, :-1] - comp[:, :, :, 1:])) \
        #           + torch.mean(torch.abs(comp[:, :, :, 1:] - comp[:, :, :, :-1])) \
        #           + torch.mean(torch.abs(comp[:, :, :-1, :] - comp[:, :, 1:, :])) \
        #           + torch.mean(torch.abs(comp[:, :, 1:, :] - comp[:, :, :-1, :]))) / 2

        # Hole Pixel Loss
        hole_loss = self.l1((1-mask) * output, (1-mask) * gt)

        # Valid Pixel Loss
        valid_loss = self.l1(mask * output, mask * gt)

        # Perceptual Loss and Style Loss
        feats_out = self.extractor(output)
        feats_comp = self.extractor(comp)
        feats_gt = self.extractor(gt)
        perc_loss = 0.0
        style_loss = 0.0
        # Calculate the L1Loss for each feature map
        for i in range(3):
            perc_loss += self.l1(feats_out[i], feats_gt[i])
            perc_loss += self.l1(feats_comp[i], feats_gt[i])
            style_loss += self.l1(gram_matrix(feats_out[i]),
                                  gram_matrix(feats_gt[i]))
            style_loss += self.l1(gram_matrix(feats_comp[i]),
                                  gram_matrix(feats_gt[i]))

        return {'valid': valid_loss,
                'hole': hole_loss,
                'perc': perc_loss,
                'style': style_loss,
                'tv': tv_loss}

# Modified block-wise feature extractor
class VGGBlockFeatureExtractor(nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, only_maxpool=False):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        vgg16.eval()
        self.normalization = Normalization(self.MEAN, self.STD)
        self.only_maxpool = only_maxpool

        if only_maxpool:
            # Extract only after max-pooling layers
            blocks_end_idx = [4, 9, 16, 23, 30]  # Indices after max-pool layers
        else:
            # Full blocks
            blocks_end_idx = [5, 10, 17, 24, 31]  # Include entire blocks
        
        features = [self.normalization]
        for i, end_idx in enumerate(blocks_end_idx):
            start_idx = blocks_end_idx[i-1] + 1 if i > 0 else 0
            block = vgg16.features[start_idx:end_idx + 1]
            features.append(block)

        self.features = nn.ModuleList(features)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = []
        for feature in self.features:
            x = feature(x)
            outputs.append(x)
        return outputs
    
class VGGFullFeatureExtractor(nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        vgg16.eval()
        self.normalization = Normalization(self.MEAN, self.STD)
        # Unpack all convolutional layers and ReLU activations, ignoring max-pooling
        features = []
        features.append(self.normalization)
        for layer in vgg16.features:
            if isinstance(layer, (nn.Conv2d, nn.ReLU)):
                features.append(layer)
        self.features = nn.ModuleList(features)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = []
        for feature in self.features:
            x = feature(x)
            outputs.append(x)
        return outputs


# # The network of extracting the feature for perceptual and style loss
# class VGG16FeatureExtractor(nn.Module):
#     """todo drop-in replace with """
#     MEAN = [0.485, 0.456, 0.406]
#     STD = [0.229, 0.224, 0.225]

#     def __init__(self):
#         super().__init__()
#         vgg16 = models.vgg16(pretrained=True)
#         vgg16.eval()
#         normalization = Normalization(self.MEAN, self.STD)
#         # Define the each feature exractor
#         self.enc_1 = nn.Sequential(normalization, *vgg16.features[:5])
#         self.enc_2 = nn.Sequential(*vgg16.features[5:10])
#         self.enc_3 = nn.Sequential(*vgg16.features[10:17])

#         # fix the encoder
#         for i in range(3):
#             for param in getattr(self, 'enc_{}'.format(i+1)).parameters():
#                 param.requires_grad = False

#     def forward(self, input):
#         feature_maps = [input]
#         for i in range(3):
#             feature_map = getattr(self, 'enc_{}'.format(i+1))(feature_maps[-1])
#             feature_maps.append(feature_map)
#         return feature_maps[1:]
    
class AlexNetFeatureExtractor(nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        alexnet = models.alexnet(pretrained=True)
        alexnet.eval()  # Set to evaluation mode
        normalization = Normalization(self.MEAN, self.STD)

        # Define each feature extractor
        self.enc_1 = nn.Sequential(normalization, *alexnet.features[:3]) # Conv + ReLU + MaxPool
        self.enc_2 = nn.Sequential(*alexnet.features[3:6]) # Conv + ReLU + MaxPool
        self.enc_3 = nn.Sequential(*alexnet.features[6:8]) # Conv + ReLU
        self.enc_4 = nn.Sequential(*alexnet.features[8:10]) # Conv + ReLU
        self.enc_5 = nn.Sequential(*alexnet.features[10:]) # Conv + ReLU + MaxPool
#         self.enc_6 = nn.Sequential(*alexnet.classifier[:2]) # Dropout + FC
#         self.enc_7 = nn.Sequential(*alexnet.classifier[2:5]) # ReLU + Dropout + FC
#         self.enc_8 = nn.Sequential(*alexnet.classifier[5:]) # ReLU + FC

        # Fix the encoder
        for i in range(5):
            for param in getattr(self, 'enc_{}'.format(i+1)).parameters():
                param.requires_grad = False

    def forward(self, input):
        feature_maps = [input]
        for i in range(5):
            feature_map = getattr(self, 'enc_{}'.format(i+1))(feature_maps[-1])
            feature_maps.append(feature_map)
        return feature_maps[1:]


# Normalization Layer for VGG
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, input):
        # normalize img
        if self.mean.type() != input.type():
            self.mean = self.mean.to(input)
            self.std = self.std.to(input)
        return (input - self.mean) / self.std


# Calcurate the Gram Matrix of feature maps
def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def dialation_holes(hole_mask):
    b, ch, h, w = hole_mask.shape
    dilation_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False).to(hole_mask)
    torch.nn.init.constant_(dilation_conv.weight, 1.0)
    with torch.no_grad():
        output_mask = dilation_conv(hole_mask)
    updated_holes = output_mask != 0
    return updated_holes.float()


def total_variation_loss(image, mask, method):
    hole_mask = 1 - mask
    dilated_holes = dialation_holes(hole_mask)
    colomns_in_Pset = dilated_holes[:, :, :, 1:] * dilated_holes[:, :, :, :-1]
    rows_in_Pset = dilated_holes[:, :, 1:, :] * dilated_holes[:, :, :-1:, :]
    if method == 'sum':
        loss = torch.sum(torch.abs(colomns_in_Pset*(
                    image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
            torch.sum(torch.abs(rows_in_Pset*(
                    image[:, :, :1, :] - image[:, :, -1:, :])))
    else:
        loss = torch.mean(torch.abs(colomns_in_Pset*(
                    image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
            torch.mean(torch.abs(rows_in_Pset*(
                    image[:, :, :1, :] - image[:, :, -1:, :])))
    return loss


if __name__ == '__main__':
    from config import get_config
    config = get_config()
    vgg = VGG16FeatureExtractor()
    criterion = InpaintingLoss(config['loss_coef'], vgg)

    img = torch.randn(1, 3, 500, 500)
    mask = torch.ones((1, 1, 500, 500))
    mask[:, :, 250:, :][:, :, :, 250:] = 0
    input = img * mask
    out = torch.randn(1, 3, 500, 500)
    loss = criterion(input, mask, out, img)
