import os
import json

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pycocotools.mask as mask_util

class BasketballtDataset(Dataset):

    def __init__(self, obj_ann_path, img_dir, split,
                 min_img_id=None, max_img_id=None, concat_img=True):
        with open(obj_ann_path) as f: 
            #! 제이슨 읽는 부분.
            anns = json.load(f)

        # search for the object id range corresponding to the image split
        min_id = 0
        if min_img_id is not None:
            while anns['image_idxs'][min_id] < min_img_id:
                min_id += 1
        max_id = len(anns['image_idxs'])
        if max_img_id is not None:
            while max_id > 0 and anns['image_idxs'][max_id - 1] >= max_img_id:
                max_id -= 1

        self.obj_masks = anns['object_masks'][min_id: max_id]
        self.img_ids = anns['image_idxs'][min_id: max_id]
        #self.cat_ids = anns['category_idxs'][min_id: max_id]
        if anns['feature_vectors'] != []: #@ feature vec을 np.array로 바꿔줌.
            self.feat_vecs = np.array(anns['feature_vectors'][min_id: max_id]).astype(float)
        else:
            self.feat_vecs = None
        self.split=split
        self.img_dir = img_dir
        self.concat_img = concat_img
        transform_list = [transforms.ToTensor()]
        self._transform = transforms.Compose(transform_list)
    def __len__(self):
        return len(self.img_ids)
    def __getitem__(self, idx):
        #! A01_AA01_T002_220916_CH01_X01_f001838.jpg
        img_name =self.img_ids[idx]+".jpg"
        img = cv2.imread(os.path.join(self.img_dir,self.split, img_name), cv2.IMREAD_COLOR)#! self.split으로 폴더 접근.
        img = self._transform(img)#@ transforms.ToTensor()
        label = -1 #@ feature vec이 없을때를 대비해서 -1로.
        if self.feat_vecs is not None:
            label = torch.Tensor(self.feat_vecs[idx])
        img_id = self.img_ids[idx]
        #cat_id = self.cat_ids[idx]
       
##########! 마스크는 박스로 불러와서 후처리. ####################################
        # mask = torch.Tensor(mask_util.decode(self.obj_masks[idx]))
       
        # seg = img.clone()
        # for i in range(3):
        #     seg[i, :, :] = img[i, :, :] * mask.float()#! 마스크 씌워주는 코드.
#!################################################################################


        transform_list = [transforms.ToPILImage(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
        if self.concat_img:
            data = img.clone().fill_(0)#! 채널 늘림. 0으로 채움.
            data[0:3] = transforms.Compose(transform_list)(seg)
            data[3:6] = transforms.Compose(transform_list)(img)
        else:
            data = img.clone().resize_.fill_(0)
            data[:, :, :] = transforms.Compose(transform_list)(seg)

        return data, label, img_id