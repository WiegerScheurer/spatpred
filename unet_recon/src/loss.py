import torch
import torch.nn as nn
from torchvision import models
from pdb import set_trace
from skimage.metrics import structural_similarity as ssim
import h5py
import pickle
import numpy as np
import os


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
        elif extractor=='vgg-conv': # All convolutional layers, this one was used for thesis. TODO: include dense layers
            self.extractor = VGGFullFeatureExtractor(only_conv=True)
        elif extractor=='vgg-conv-dense': # Both the convolutional layers and the dense layers
            self.extractor = VGGFullFeatureExtractor(only_conv=True, include_dense=True)
        elif extractor=='vgg-dense': # Both the convolutional layers and the dense layers
            self.extractor = VGGFullFeatureExtractor(only_conv=False, include_dense=True)
        elif extractor=='vgg-b':
            self.extractor=VGGBlockFeatureExtractor()
        elif extractor=='vgg-bp':
            self.extractor=VGGBlockFeatureExtractor(only_maxpool=True)
        elif extractor=='vgg-bn':
            self.extractor=VGGBlockFeatureExtractor(batchnorm=True)
        elif extractor=='vgg-bn-full':
            self.extractor=VGGFullFeatureExtractorCustom(only_conv=False, only_relu=False)
        elif extractor=='vgg-bn-sel-conv':
            self.extractor=VGGFullFeatureExtractorCustom(only_sel_conv=True)
        elif extractor=='vgg-bn-relu':
            self.extractor=VGGFullFeatureExtractorCustom(only_relu=True)
        elif extractor=='vgg-bn-mp':
            self.extractor=VGGFullFeatureExtractorCustom(only_mp=True)
        else:
            raise ValueError('expecting "vgg" or "alex"')


    def forward(self, input, mask, output, gt, save_gt_featmaps:bool=False):
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
        # ADD PRINT STATEMENT HERE TO CHECK WHETHER IT IS THE FULL OR CROPPPED
        # THEN INCLUDE CONDITIONAL TO CHECK AND SAVE FILE ACCORDINGLY
        # It does it twice because one is for L1, the other for MSE, but I only need 1 
        # print(f'this is the size of the groundtruth patch: {gt.shape}')
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
                        
            # # DEBUGGERING::
            # print(f'feats_comp[{layer_i}] shape: {feats_comp[layer_i].shape}')
            # print(f'feats_gt[{layer_i}] shape: {feats_gt[layer_i].shape}')
            
            # loss_dict[f'content_loss_{layer_i}']=(
            #     self.loss(feats_comp[layer_i], feats_gt[layer_i]).mean(dim=(1,2,3)).numpy()) 
            # loss_dict[f'style_loss_{layer_i}']=(
            #     self.loss(gram_matrix(feats_comp[layer_i]),
            #               gram_matrix(feats_gt[layer_i])).mean(dim=(1,2)).numpy())
            
            if len(feats_comp[layer_i].shape) == 4:
                loss_dict[f'content_loss_{layer_i}'] = (
                    self.loss(feats_comp[layer_i], feats_gt[layer_i]).mean(dim=(1,2,3)).numpy())
                loss_dict[f'style_loss_{layer_i}'] = (
                    self.loss(gram_matrix(feats_comp[layer_i]),
                            gram_matrix(feats_gt[layer_i])).mean(dim=(1,2)).numpy())
            elif len(feats_comp[layer_i].shape) == 2:
                loss_dict[f'content_loss_{layer_i}'] = (
                    self.loss(feats_comp[layer_i], feats_gt[layer_i]).mean(dim=1).numpy())
                loss_dict[f'style_loss_{layer_i}'] = (
                    self.loss(gram_matrix(feats_comp[layer_i]),
                            gram_matrix(feats_gt[layer_i])).mean(dim=1).numpy())
    
        if self.add_loss_suff:
            loss_dict = {key+f'_{self.loss_str}': value for key, value in loss_dict.items()}

        if save_gt_featmaps:
            # feats_comp_np = [feat.detach().cpu().numpy() for feat in feats_comp] # WADDITION
            feats_gt_np = [feat.detach().cpu().numpy() for feat in feats_gt] # WADDITION
            
            if self.loss_str == 'L1':

                filename = '/home/rfpred/data/custom_files/subj01/pred/featmaps/feats_gt_npTESTERITUS.pkl'
                base_filename, ext = os.path.splitext(filename)
                i = 0

                while os.path.exists(filename):
                    i += 1
                    filename = f"{base_filename}_{i}{ext}"

                with open(filename, 'wb') as f:
                    pickle.dump(feats_gt_np, f)
            
        
        # with open('/home/rfpred/data/custom_files/subj01/pred/featmaps/feats_gt_np.pkl', 'wb') as f:
        #     pickle.dump(feats_gt_np, f)
        # featmap_dict = {'feats_comp': feats_comp_np, 'feats_gt': feats_gt_np} # WADDITION


        # with h5py.File('TESTTESTfeatmap_dict.h5', 'w') as hf:
            # for key in featmap_dict.keys():
                # hf.create_dataset(key, data=featmap_dict[key])

        # return loss_dict, featmap_dict # WADDITION

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

    def __init__(self, only_maxpool=False, batchnorm:bool=False):
        super().__init__()
        vgg16 = models.vgg16_bn(pretrained=True) if batchnorm else models.vgg16(pretrained=True)    
        vgg16.eval()
        self.normalization = Normalization(self.MEAN, self.STD)
        self.only_maxpool = only_maxpool

        if only_maxpool:
            # Extract only after max-pooling layers
            blocks_end_idx = [4, 9, 16, 23, 30]  # Indices after max-pool layers
        elif batchnorm:
            # Extract the blocks after convolutional layers from vgg-16 with batchnorm (vgg16_bn)
            blocks_end_idx = [7, 14, 24, 34, 40]
        else:
            # Full blocks
            blocks_end_idx = [5, 10, 17, 24, 31]  # Include entire blocks
        
        features = [self.normalization]
        for i, end_idx in enumerate(blocks_end_idx):
            start_idx = blocks_end_idx[i-1] + 1 if i > 0 else 0
            block = vgg16.features[start_idx:end_idx + 1]
            features.append(block) # Hierdoor zijn het er 6

        self.features = nn.ModuleList(features)
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = []
        for feature in self.features:
            x = feature(x)
            outputs.append(x)
        return outputs
    
# class VGGFullFeatureExtractor(nn.Module):
#     MEAN = [0.485, 0.456, 0.406]
#     STD = [0.229, 0.224, 0.225]

#     def __init__(self, only_conv:bool = False):
#         super().__init__()
#         vgg16 = models.vgg16(pretrained=True)
#         vgg16.eval()
#         self.normalization = Normalization(self.MEAN, self.STD)
#         # Unpack all convolutional layers and ReLU activations, ignoring max-pooling
#         features = []
#         features.append(self.normalization)
#         for layer in vgg16.features:
#             if only_conv:
#                 if isinstance(layer, nn.Conv2d):
#                     features.append(layer)
#             else:     
#                 if isinstance(layer, (nn.Conv2d, nn.ReLU)):
#                     features.append(layer)
#         self.features = nn.ModuleList(features)
#         for param in self.features.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         outputs = []
#         for feature in self.features:
#             x = feature(x)
#             outputs.append(x)
#         return outputs
    
    
class VGGFullFeatureExtractor(nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, only_conv:bool = False, include_dense:bool = False):
        super().__init__()
        self.only_conv = only_conv
        self.include_dense = include_dense
        vgg16 = models.vgg16(pretrained=True)
        vgg16.eval()
        self.normalization = Normalization(self.MEAN, self.STD)
        # Unpack all convolutional layers and ReLU activations, ignoring max-pooling
        features = []
        features.append(self.normalization)
        for layer in vgg16.features:
            if only_conv:
                if isinstance(layer, nn.Conv2d):
                    features.append(layer)
            else:     
                if isinstance(layer, (nn.Conv2d, nn.ReLU)):
                    features.append(layer)
        self.features = nn.ModuleList(features)
        # Include dense layers if include_dense is True
        if include_dense:
            self.classifier = vgg16.classifier
        for param in self.parameters():
            param.requires_grad = False
        # Add an AdaptiveAvgPool2d layer
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        outputs = []
        for feature in self.features:
            x = feature(x)
            if self.only_conv:
                outputs.append(x)
        # print(x.size())  # Print the size of the tensor after the convolutional layers #DEBUG OPTION
        # Apply the AdaptiveAvgPool2d layer
        x = self.pool(x)
        # print(x.size())  # Print the size of the tensor after pooling #DEBUG OPTION
        # Process through dense layers if they are included
        if hasattr(self, 'classifier'):
            x = x.view(x.size(0), -1)  # Flatten the tensor
            # print(x.size())  # Print the size of the tensor after flattening #DEBUG OPTION
            for layer in self.classifier:
                x = layer(x)
                if isinstance(layer, nn.Linear) and self.include_dense:
                    outputs.append(x)
        return outputs


# THIS ONE WORKED NICELY, but couldn't do only dense
    # def forward(self, x):
    #     outputs = []
    #     for feature in self.features:
    #         x = feature(x)
    #         outputs.append(x)
    #     print(x.size())  # Print the size of the tensor after the convolutional layers
    #     # Apply the AdaptiveAvgPool2d layer
    #     x = self.pool(x)
    #     print(x.size())  # Print the size of the tensor after pooling
    #     # Process through dense layers if they are included
    #     if hasattr(self, 'classifier'):
    #         x = x.view(x.size(0), -1)  # Flatten the tensor
    #         print(x.size())  # Print the size of the tensor after flattening
    #         for layer in self.classifier:
    #             x = layer(x)
    #             if isinstance(layer, nn.Linear):
    #                 outputs.append(x)
    #     return outputs
        
# I haven't changed it yet, so I don't know whether it'll work. I'm still running the convolutional layer computations of the peripheral patches.
# Potential problem is the different dimension of the dense layers, as they're flat. This implementation might not be ideal.
    
    
    
class VGGFullFeatureExtractorCustom(nn.Module):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, only_conv:bool = False, only_relu:bool = False, only_sel_conv:bool = False):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        vgg16.eval()
        self.normalization = Normalization(self.MEAN, self.STD)
        # Unpack all convolutional layers and ReLU activations, ignoring max-pooling
        features = []
        # features.append(self.normalization)
        selected_indices = [0, 2, 5, 10, 17, 21, 24, 28]

        for i, layer in enumerate(vgg16.features):
            if only_sel_conv:
                if isinstance(layer, nn.Conv2d) and i in selected_indices:
                    features.append(layer)
            elif only_relu:
                if isinstance(layer, nn.ReLU):
                    features.append(layer)
            else:     
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
        # self.enc_1 = nn.Sequential(normalization, *alexnet.features[:2]) # Conv + ReLU + MaxPool
        self.enc_2 = nn.Sequential(*alexnet.features[3:5]) # Conv + ReLU (OG is tot 6, dus met maxpool)
        # self.enc_2 = nn.Sequential(*alexnet.features[3:5]) # Conv + ReLU + MaxPool
        self.enc_3 = nn.Sequential(*alexnet.features[6:8]) # Conv + ReLU
        self.enc_4 = nn.Sequential(*alexnet.features[8:10]) # Conv + ReLU
        self.enc_5 = nn.Sequential(*alexnet.features[10:12]) # Conv + ReLU (OG is tot einde, dus met maxpool)
        # self.enc_5 = nn.Sequential(*alexnet.features[10:11]) # Conv + ReLU + MaxPool
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


# ORIGINAL:::::
# # Calcurate the Gram Matrix of feature maps
# def gram_matrix(feat):
#     (b, ch, h, w) = feat.size()
#     feat = feat.view(b, ch, h * w)
#     feat_t = feat.transpose(1, 2)
#     gram = torch.bmm(feat, feat_t) / (ch * h * w)
#     return gram

def gram_matrix(feat):
    if len(feat.shape) == 4:
        (b, ch, h, w) = feat.size()
        feat = feat.view(b, ch, h * w)
        feat_t = feat.transpose(1, 2)
        gram = torch.bmm(feat, feat_t) / (ch * h * w)
    elif len(feat.shape) == 2:
        (b, ch) = feat.size()
        feat_t = feat.transpose(0, 1)
        gram = torch.mm(feat, feat_t) / ch
    else:
        raise ValueError(f'Expected input tensor to have 2 or 4 dimensions, but got {len(feat.shape)} dimensions')
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

