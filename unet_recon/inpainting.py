from PIL import Image
import torch
import torch.nn.functional as nnF
import torchvision.transforms.functional as TF
from torchvision.ops import masks_to_boxes
# from torchvision
from .src.model import PConvUNet
from .src.loss import ReconLoss as NNLoss
# from lpips import LPIPS
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity as ssim

from pdb import set_trace

MODEL_LOC =Path(__file__).parent / 'pretrained_models'
# MODEL_LOC= Path('/project/3018039.02/vispred/partialconv_pytorch2/pretrained_models')

class UNet():
    """Load pretrained UNet partialconv architecture for image inpainting (Liu et al. ECCV, 2018). 
    Main method: analyse_images, recon_imgs
    
    feature_model, can be ('alex','vgg','vgg-b', 'vgg-bp' (only the max-pool layers of each block))
    
    
    ---------------------------------------------------------------------------------------------
    Note, this is just a superclas built on top of PConvUNet class, from:
    https://github.com/tanimutomo/partialconv 

    ... in turn based on the official implementation: 
    - https://github.com/NVIDIA/partialconv
    and on: 
    - https://github.com/MathiasGruber/PConv-Keras
    - https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/
    
    --------------
    
    """
    def __init__(self, checkpoint_name='pconv_circ-places60k-fine.pth',device='auto',im_size=(3,256,256),
                 resize_method='zero_padd',feature_model='alex',loss='L1',add_loss_suff=True):
        
        # Define the used device
        if device=='auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        else:
            self.device=device

        # Define the model
        print("Loading the Model...")
        print("kommetje soep")
        self.model = PConvUNet(finetune=False, layer_size=7)
        self.model.load_state_dict(torch.load(MODEL_LOC/checkpoint_name, 
                                              map_location=torch.device(device=self.device))['model'])
        self.model.to(self.device)
        self.model.eval()
        
        # define perceptual loss model        
        self.loss_obj_l1=NNLoss(extractor=feature_model,loss='L1',add_loss_suff=add_loss_suff)[0] # WADDITION
        self.loss_obj_l2=NNLoss(extractor=feature_model,loss='MSE',add_loss_suff=add_loss_suff)[0] # WADDITION
        # self.loss_obj_l1=NNLoss(extractor=feature_model,loss='L1',add_loss_suff=add_loss_suff)
        # self.loss_obj_l2=NNLoss(extractor=feature_model,loss='MSE',add_loss_suff=add_loss_suff)
        
        self.featmaps=NNLoss(extractor=feature_model,loss='L1',add_loss_suff=add_loss_suff)[1] # WADDITION

        # store additional attributes
        self._im_size=im_size # todo: can we handle arbitrary aspect ratios w/o padding?
        self._max_size=max(self._im_size)
        self._resize_method=resize_method
        
    def analyse_images(self,imgs_in,mask_in,eval_mask=None,return_recons=False):
        """
        main method for inpainting-based predictability analysis. 
        from a range of images and (range of) mask(s), comptus local spatial predictability of the 
        image patch (RF) inside the mask. 
        
        In: 
        
        - imgs_in: (list of) PIL.Images
            Ground truth images. Must have same size mask image(s)
        - mask_in: (list of) PIL.Images of masks.
            If a list, must be same length as imgs_in.
            If a single mask image, same mask is applied to all imgs
        -eval_mask: None OR nd.array mask OR PIL.Image (Default: None)
            OPTIONAL: single rectangular mask that spans the inpainting mask, to evaluate
            the similarity between the inpainted and ground truth image patch locally, instead of
            across the entire image. Note that while the method accepts  _different_ inpainting masks
            for each input image, it only accept a _single_ evaluation mask, applied to each image. 
       - return_recons: Bool (Default: False)
           only return summary statistics or also return the reconstructions themselves?
        
        Out:
        - eval_dict:
            Dictionary with keys:
            -pixel_corr: pixelwise pearson correlation
            -lpips:      learned perceptual image patch similarity 
            -lpips_perlayer: L2 loss per layer. (metrics should be standardised per layer, due to scaling cannot 
                             be numerically compared between layers). 
                    
        """
        to_np=lambda x: np.array(TF.to_pil_image(x))

        # first we get an iteration 
        recon_pld=self.recon_imgs(imgs_in,mask_in,remove_padding=True)

        if eval_mask is not None:
            recon_pld=self._apply_eval_mask(recon_pld,eval_mask)
#         self.loss_obj=NNLoss()
        
        # compute NN losses 
#         loss_dict=self.loss_obj.forward(recon_pld['input_masked'],recon_pld['masks'],
#                                          recon_pld['out_raw'],recon_pld['input_gt'])
        
         # compute perceptual loss
        with torch.no_grad():
            loss_dict_1=self.loss_obj_l1.forward(recon_pld['input_masked'],recon_pld['masks'],
                                                 recon_pld['out_raw'],recon_pld['input_gt'])
            loss_dict_2=self.loss_obj_l2.forward(recon_pld['input_masked'],recon_pld['masks'],
                                            recon_pld['out_raw'],recon_pld['input_gt'])

            ssims=np.array([ssim(to_np(recon_pld['input_gt'][img_ix]),
                         to_np(recon_pld['out_composite'][img_ix]),multichannel=True, channel_axis = 2)
                    for img_ix in range(recon_pld['out_composite'].shape[0])])

            eval_pld={
                      'ssim':ssims
                     }

            eval_pld.update(loss_dict_1)
            eval_pld.update(loss_dict_2)
            
            self.featmaps = {'featmaps': self.featmaps}
            # self.featmaps = {f'featmaps_{i}': featmap for i, featmap in enumerate(self.featmaps)}
            eval_pld.update(self.featmaps) # WADDITION

#         if normalise: # undo the normalisation
#             recon_pld={k:TF.normalize(v,mean=(-1,-1,-1),std=(2,2,2)) for k,v in recon_pld.items()}
        # 
        if return_recons: eval_pld['recon_dict']=recon_pld
        
        # eval_pld['featmaps'] = self.featmaps
        # return eval_pld, kont
        return(eval_pld)
                    
    def recon_imgs(self,imgs_in,mask_in,remove_padding=False):
        """for list of PIL images and (list of) pil mask, generate reconstructions
        in:
        - imgs_in: (list of) PIL.Images
            Ground truth images. Must have same size mask image(s)
        - mask_in: (list of) PIL.Images of masks.
            If a list, must be same length as imgs_in.
            If a single mask image, same mask is applied to all imgs
        - remove_padding: bool -- Default=False
            if images are padded to force aspect ratio 1, remove padding in output?
        
        out: 
        - recon_pld: dict (payload)
            with keys:
                - out_raw: complete output
                - out_composite: mixture of input plus the filled in output
                - input_masked : input with mask zeroed out. 
                - input_gt:      input ground truth
                - masks:         masks
            all are Torch.Tensor objects with shape (n_imgs,3,n_rows,n_cols)
        """
        
        # convert (list of) PIL to tensor and make sure same size and aspect ratio
        imgs_t,masks_t,padd_info=self._parse_im_masks(imgs_in,mask_in)
        inp = imgs_t * masks_t # mask out info from input image 
        
        with torch.no_grad():
            input_=inp.to(self.device)   # move to gpu
            mask_=masks_t.to(self.device)
            raw_out, _ = self.model(input_, mask_)

        # post-process 
        raw_out=raw_out.to(torch.device('cpu')).clamp(0.0, 1.0)
        out_composite= masks_t * inp + (1 - masks_t) * raw_out
        
        # package into output 
        recon_pld={'out_raw':raw_out,'out_composite':out_composite,
                   'input_masked':inp,'input_gt':imgs_t,'masks':masks_t}
        
        # in case input was (zero) padded, remove padding to restore aspect ratio?
        if (padd_info is not None) and remove_padding==True:
            recon_pld=self._unpadd_recon_pld(recon_pld,padd_info)

        return(recon_pld)
    
    def _parse_im_masks(self,imgs_in,mask_in):
        """internal helper function.
        from (list of) PIL images and (list of) PIL mask, get Tensors of correct shapes to process.
        
        """
        # check if masks and images are lists or PIL Image instances
        assert isinstance(mask_in,(list,Image.Image)), 'mask must be (list of) PIL image(s)'
        assert isinstance(imgs_in,(list,Image.Image)), 'image must be (list of) PIL image(s)'
        
        # in case img or mask is single Image, wrap into lists (and ensure mask is same length as images)
        if isinstance(imgs_in,Image.Image): imgs_in=[imgs_in]
        if isinstance(mask_in,Image.Image): mask_in=[mask_in]*len(imgs_in)
        
        # to be sure, assert all images and masks have same size
        assert len(set([(this_im.size) for this_im in imgs_in]))==1, "all images must have same size"
        assert len(set([(this_mask.size) for this_mask in mask_in]))==1, "all masks must have same size"
        assert imgs_in[0].size==mask_in[0].size, 'images and masks sizes must match'
        
        # if we need to resize, resize the full list 
        if np.max(imgs_in[0].size)>self._max_size: 
            imgs_in=self._resize_pil_img(imgs_in) 
            mask_in=self._resize_pil_img(mask_in,interp=Image.NEAREST) 
        
        
        # convert to tensor (and ensure we have 3 colour channels)
        imgs_t=[TF.to_tensor(img.convert('RGB')) for img in imgs_in]
        masks_t=[TF.to_tensor(msk.convert('RGB')) for msk in mask_in]
        
        # if aspect ratio is not 1, do zero/1 padding
        if (imgs_in[0].size[0]/imgs_in[0].size[1])!=1:
            # apply zero/1 padding, and store the dimensions where we have padded (to revert later)
            padd_dim_info=tuple(np.repeat((np.array(self._im_size)-imgs_t[0].shape)/2,2).astype(int))
            imgs_t=[self._padd_tensor_img(im_t,desired_shape=self._im_size) for im_t in imgs_t]
            masks_t=[self._padd_tensor_img(msk_t,desired_shape=self._im_size,padd_val=1) for msk_t in masks_t]
        else:
            padd_dim_info=None

        # create 4d tensor, and create composite input 
        imgs_t=torch.stack(imgs_t)
        masks_t=torch.stack(masks_t)
        return(imgs_t,masks_t,padd_dim_info)

    def _resize_pil_img(self,pil_imgs_in:list,interp=Image.BILINEAR):
        """resize the list of PIL images. assumes all have same size.
        target size is established such that dimensions will match certain maximum size (e.g. 256)
        """
        resize_fact=self._max_size/np.max(pil_imgs_in[0].size)
        trg_size=tuple(np.round(resize_fact*np.array(pil_imgs_in[0].size)).astype(int))
        return([this_img.resize(trg_size,resample=interp) for this_img in pil_imgs_in])
    
    def _padd_tensor_img(self,tens_in,desired_shape=(3,256,256),padd_val=0):
        """padd incoming tensor with zero-padding
        in: 
        -tens_in: Tensor
            incoming image [colour,rows,cols]
        - desired_shape: tuple, Default: (3,256,256)
            desired shape
        - padd_val: float/int, Default: 0
            padd value 

        out:
        -tensor of shape desired_shape
        """
        p3d=tuple(np.flip(np.repeat((np.array(desired_shape)-tens_in.shape)/2,2).astype(int)))
        return(nnF.pad(tens_in, p3d, "constant", padd_val))  # zero/1 padding
    
    def _unpadd_recon_pld(self,pld_in,padd_info):
        """From a dict with tensors as returned by recon_imgs, undo the padding so as to restore
        olriginal aspect ratio. """
        # we assume symmetric pading, but the thing below can also deal with assymetric padding.
        dim2padd=np.unique(np.floor(np.nonzero(padd_info)[0]/2))

        if (len(dim2padd)>=2) or (dim2padd==0):
            raise ValueError(f'expected to (un)padd height OR width... info in: {padd_info}', )
        elif dim2padd==1:
            pld_out={k:v[:,:,padd_info[2]:-padd_info[3],:] for k,v in pld_in.items()}
        elif dim2padd==2:
            pld_out={k:v[:,:,padd_info[4]:-padd_info[5],:] for k,v in pld_in.items()}
            
        return(pld_out)
    
    def _apply_eval_mask(self,recon_pld_in,eval_mask_in):
        """given a recon_pld (reconstruction dict) and an evaluation mask, return masked recon_pld
        
        ... 
        
        """
        assert isinstance(eval_mask_in,(np.ndarray,Image.Image)), "eval mask must be array or PIL.Image!"
        if isinstance(eval_mask_in,np.ndarray): eval_mask_in=Image.fromarray(eval_mask_in)

        # make sure size is correct etc 
        desired_msk_size=tuple(np.flip(recon_pld_in['out_raw'].shape[-2:])) # in width,height coordinates
        if eval_mask_in.size!=desired_msk_size:
            eval_mask_in=eval_mask_in.resize(desired_msk_size,Image.NEAREST)

        xmin,ymin,xmax,ymax=masks_to_boxes(
            TF.to_tensor(eval_mask_in)).squeeze().squeeze().numpy().astype(int)

        recon_pld_out={tensor_k:tensor_v[:,:,ymin:ymax,xmin:xmax] 
                       for tensor_k,tensor_v in recon_pld_in.items()}
        return(recon_pld_out)

    
def _corr_metric(a,b):
    """compute pearson correlation between _columns_ of two arrays"""
    zs = lambda v: (v-v.mean(0))/v.std(0) ## z-score 
    return((zs(a)*zs(b)).mean(0)) 
