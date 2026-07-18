import os
import numpy as np
from scipy.ndimage import zoom
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import Dataset
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from detectron2.data import detection_utils as utils
import sys
sys.path.append(r'/space1/liuchang/VI/SegMiF-main/MaskFormer/detectron2')
from projects.PointRend.point_rend import ColorAugSSDTransform
sys.path.append(r"/space1/liuchang/VI/SegMiF-main/segment-anything")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import time
from torchvision.utils import save_image

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    msk = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = [np.random.random(3) * 255]
        msk[m] = color_mask
    return msk

def probs_process(anns):
    ps = [ann['probs'] for ann in anns]
    ps = np.stack(ps, axis=2)
    return ps

def save_img(image, image_vi, sem_seg_gt, sam_masks, sam_masks_vi):
    save_img = Image.fromarray(image)
    save_img_vi = Image.fromarray(image_vi)
    save_label = Image.fromarray((sem_seg_gt*255).astype(np.uint8))        
    save_mask = show_anns(sam_masks)
    save_mask = Image.fromarray((save_mask).astype(np.uint8))
    save_mask_vi = show_anns(sam_masks_vi)
    save_mask_vi = Image.fromarray((save_mask_vi).astype(np.uint8))
    
    save_img.save("/space0/liuchang/SegMiF/SAM_mask_tokens/crop.png")
    save_img_vi.save("/space0/liuchang/SegMiF/SAM_mask_tokens/crop_vi.png")
    save_label.save("/space0/liuchang/SegMiF/SAM_mask_tokens/crop_label.png")
    save_mask.save("/space0/liuchang/SegMiF/SAM_mask_tokens/crop_mask.png")
    save_mask_vi.save("/space0/liuchang/SegMiF/SAM_mask_tokens/crop_mask_vi.png")

    
def save_all(image, sem_seg_gt, ps):
    save_img = Image.fromarray(image)
    save_label = Image.fromarray((sem_seg_gt*255).astype(np.uint8))        
    ps = ps.unsqueeze(1)  # 在通道维度上添加一个维度
    save_image(ps, "/space0/liuchang/SegMiF/SAM_mask_tokens/crop_mask1.png", nrow=10, padding=2)
    save_img.save("/space0/liuchang/SegMiF/SAM_mask_tokens/crop1.png")
    save_label.save("/space0/liuchang/SegMiF/SAM_mask_tokens/crop_label1.png")

def load_sam(points_per_side):
        # load model
        sam_checkpoint = r"/space0/liuchang/SegMiF/pretrained/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        for param in sam.parameters():
            param.requires_grad = False
        mask_generator = SamAutomaticMaskGenerator(
            model=sam, 
            points_per_side=points_per_side, 
            pred_iou_thresh=0.86, 
            stability_score_thresh=0.92, 
            crop_n_layers=0, 
            crop_n_points_downscale_factor=1, 
            min_mask_region_area=0,  # Requires open-cv to run post-processing
            )
        return mask_generator
    
class  MSRSDataset(Dataset):
    def __init__(self, root_dir=r'/space1/liuchang/VI/SegMiF-main/MSRS_Dataset/test', is_train= True):
        self.root_dir = root_dir
        self.image_vi_dir = os.path.join(root_dir, 'Visible') # Infrared
        self.image_dir = os.path.join(root_dir, 'IRwarp') # Infrared
        self.label_dir = os.path.join(root_dir, "Label")
        self.is_train = is_train
        if is_train== True:
            self.augs0 = T.AugmentationList([
                                            T.ResizeShortestEdge([360, 450, 540, 630, 720], 2560, "choice"),
                                            T.RandomCrop_CategoryAreaConstraint(crop_type="absolute", crop_size = [360, 640]),
                                            ColorAugSSDTransform(img_format= "RGB"),
                                            T.RandomFlip(),
                                            ]
                                            )
        
        # read image
        self.img_arr_list = []
        self.img_vi_arr_list = []
        self.label_arr_list = []
        self.ps_list = []
        self.sam_mask_list = []
        self.name_list = []
        self.filenames = sorted(os.listdir(self.image_dir))
        length = len(self.filenames)

        for i in tqdm(range(length)):
            image_path = os.path.join(self.image_dir, self.filenames[i])      # ir
            image_vi_path = os.path.join(self.image_vi_dir, self.filenames[i])   # vi
            label_path = os.path.join(self.label_dir, self.filenames[i])      # label
            
            image = utils.read_image(image_path, format="RGB")
            image_vi = utils.read_image(image_vi_path, format="RGB")
            sem_seg_gt = utils.read_image(label_path)
            self.img_arr_list.append(image)
            self.img_vi_arr_list.append(image_vi)
            self.label_arr_list.append(sem_seg_gt)
    
    
    def __getitem__(self, idx):

        # load input
        image = self.img_arr_list[idx]         # ir
        image_vi = self.img_vi_arr_list[idx]   # vi
        sem_seg_gt = self.label_arr_list[idx]  # label
        filename = self.filenames[idx]

        # Data enhancement
        aug_input = T.AugInput(image, image_vi, sem_seg=sem_seg_gt)
        if self.is_train == True:
            transforms = self.augs0(aug_input)
        image_ = aug_input.image
        image_vi_ = aug_input.image_vi
        sem_seg_gt_ = aug_input.sem_seg

        im_size = image_.shape[:2]

        # Convert PIL images to tensors
        image = torch.tensor(np.array(image_)).permute(2, 0, 1) # 3, 480, 640
        image_vi = torch.tensor(np.array(image_vi_)).permute(2, 0, 1) # 3, 480, 640
        sem_seg_gt = torch.tensor(np.array(sem_seg_gt_))      

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(256)
            classes = np.unique(sem_seg_gt)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            
            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]))
                instances.gt_masks = masks.tensor

        data = {
            "image": image,
            "image_vi": image_vi,
            "sem_seg_gt": sem_seg_gt,
            # "sam_masks":  ps,
            # "sam_masks_vi":  ps_vi,
            "classes": classes,
            "instances": instances,
            "height": image.shape[1],
            "width":image.shape[2],
            "im_size": im_size,
            "file_name":filename,
        }
        torch.cuda.empty_cache()
        return data
    
    def __len__(self):
        return len(self.filenames)
        # length = 18
        # return length

if __name__ == '__main__':
    test_obj = MSRSDataset()
    t = 0
    for i in range(12):
        start = time.time()
        data = test_obj[i]
        end = time.time()
        t += end - start
    print("times:", t)