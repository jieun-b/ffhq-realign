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

import os, sys
import argparse
from tqdm import tqdm
import torch
from PIL import PngImagePlugin

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
from decalib.utils.face_alignment import recreate_aligned_images

##### check the gcc version and modify utils/renderer.py


def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)

    for subdir in os.listdir(args.inputpath):
        
        dst_subdir = os.path.join(savefolder, subdir)
        os.makedirs(dst_subdir, exist_ok=True)

        inputpath = os.path.join(args.inputpath, subdir)

        # load images 
        testdata = datasets.TestData(inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)
        
        # for i in range(len(testdata)):
        for i in tqdm(range(len(testdata))):
            name = testdata[i]['imagename']
            images = testdata[i]['image'].to(device)[None,...]
            with torch.no_grad():
                codedict = deca.encode(images)
                ##### neutral expression
                codedict['exp'] = torch.zeros_like(codedict['exp'])
                codedict['pose'][:,3:] = torch.zeros_like(codedict['pose'][:,3:])

                ##### transform
                tform = testdata[i]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to(device)
                original_image = testdata[i]['original_image'][None, ...]

                ##### decode
                batch_size = images.shape[0]

                verts, landmarks2d, landmarks3d = deca.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
                if deca.cfg.model.use_tex:
                    albedo = deca.flametex(codedict['tex'])
                else:
                    albedo = torch.zeros([batch_size, 3, deca.uv_size, deca.uv_size], device=images.device) 
                landmarks3d_world = landmarks3d.clone() 

                ## projection
                landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]#; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
                landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']); landmarks3d[:,:,1:] = -landmarks3d[:,:,1:] #; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
                trans_verts = util.batch_orth_proj(verts, codedict['cam']); trans_verts[:,:,1:] = -trans_verts[:,:,1:]
                opdict = {
                    'verts': verts,
                    'trans_verts': trans_verts,
                    'landmarks2d': landmarks2d,
                    'landmarks3d': landmarks3d,
                    'landmarks3d_world': landmarks3d_world,
                }

                points_scale = [deca.image_size, deca.image_size]
                _, _, h, w = original_image.shape
                # import ipdb; ipdb.set_trace()
                trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])
                landmarks2d = transform_points(landmarks2d, tform, points_scale, [h, w])
                landmarks3d = transform_points(landmarks3d, tform, points_scale, [h, w])
                opdict['landmarks2d'] = landmarks2d
                opdict['landmarks3d'] = landmarks3d

            ##### save landmark
            original_image = original_image.cpu().numpy()
            predicted_landmark = opdict['landmarks2d'][0].cpu().numpy()
            image = original_image[0]
            image = image.transpose(1,2,0)[:,:,[2,1,0]].copy(); image = (image*255)
            predicted_landmark[...,0] = predicted_landmark[...,0]*image.shape[1]/2 + image.shape[1]/2
            predicted_landmark[...,1] = predicted_landmark[...,1]*image.shape[0]/2 + image.shape[0]/2
            
            # realign and crop
            recreate_aligned_images(name, os.path.join(inputpath, name+'.png'), dst_subdir, predicted_landmark, output_size=args.sample_size)
            
    print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='ffhq/in-the-wild-images', type=str,
                        help='path to the test data directory')
    parser.add_argument('-s', '--savefolder', default='ffhq/results', type=str,
                        help='path to the output directory')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    parser.add_argument('--sample_size', default=224, type=int,
                        help='output size for aligned/cropped images')
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    main(parser.parse_args())