# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from glob import glob

from .detector import MediaPipe

def video2sequence(video_path, sample_step=10):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        # if count%sample_step == 0:
        imagepath = os.path.join(videofolder, f'{video_name}_frame{count:04d}.jpg')
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='mediapipe', sample_step=10, json_path=None):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath): 
            self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            self.imagepath_list = video2sequence(testpath, sample_step)
        else:
            print(f'please check the test path: {testpath}')
            exit()
        # print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        
        self.annotation_dict = {}
        if json_path is not None and os.path.exists(json_path):
            with open(json_path, "r") as f:
                annotation = json.load(f)
            self.annotation_dict = {
                str(k).zfill(5): v["in_the_wild"]["face_landmarks"]
                for k, v in annotation.items()
                if "in_the_wild" in v and "face_landmarks" in v["in_the_wild"]
            }

        if face_detector == 'mediapipe':
            self.face_detector = MediaPipe()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        h, w, _ = image.shape
        if self.iscrop:
            if imagename in self.annotation_dict:
                kpt = np.array(self.annotation_dict[imagename])  # (68, 2)
                left, right = np.min(kpt[:,0]), np.max(kpt[:,0])
                top, bottom = np.min(kpt[:,1]), np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            else:
                # fallback: mediapipe
                print(f"[INFO] using mediapipe for face detection: {imagename}")
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 4:
                    print(f"[SKIP] no face detected in {imagepath}")
                    return None
                left, right = bbox[0], bbox[2]
                top, bottom = bbox[1], bbox[3]
                old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.
        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        
        # if imagename in self.annotation_dict:
        #     kpt = np.array(self.annotation_dict[imagename])  # (68, 2)

        #     debug_img = (image * 255).astype(np.uint8).copy()  # 원본 스케일 복원
        #     for (x, y) in kpt.astype(np.int32):
        #         cv2.circle(debug_img, (x, y), 2, (0, 255, 0), -1)

        #     os.makedirs("debug_outputs", exist_ok=True)
        #     cv2.imwrite(
        #         os.path.join("debug_outputs", f"{imagename}_orig_landmarks.png"),
        #         cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
        #     )
        
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                'tform': torch.tensor(tform.params).float(),
                'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }