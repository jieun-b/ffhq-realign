import os, sys
import cv2
import argparse
import numpy as np
import torch
import gc
import glob
from tqdm import tqdm
from PIL import PngImagePlugin
from concurrent.futures import ThreadPoolExecutor

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.deca import DECA
from src.datasets import datasets 
from src.utils.config import cfg as deca_cfg
from src.face_alignment import recreate_aligned_images

##### check the gcc version and modify utils/renderer.py
    
def main(args):
    executor = ThreadPoolExecutor(max_workers=args.num_workers)
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)

    # load json annotation
    anno = None
    if args.jsonpath is not None and os.path.isfile(args.jsonpath):
        import json
        with open(args.jsonpath, "r") as f:
            anno = json.load(f)

    input_files = sorted(glob.glob(os.path.join(args.inputpath, "*.png")))
    existing = {os.path.splitext(f)[0] for f in os.listdir(savefolder) if f.endswith(".png")}
    input_list = [
        f for f in input_files
        if os.path.splitext(os.path.basename(f))[0] not in existing
    ]
    # load images
    testdata = datasets.TestData(
        input_list,
        iscrop=args.iscrop,
        face_detector=args.detector,
        sample_step=args.sample_step,
        json_path=args.jsonpath
    )

    for i in tqdm(range(len(testdata))):
        sample = testdata[i]
        if sample is None:
            continue

        name = os.path.splitext(os.path.basename(sample['imagename']))[0]
        images = sample['image'].to(device)[None, ...]

        with torch.no_grad():
            codedict = deca.encode(images)
            codedict_neutral = {k: v.clone() for k, v in codedict.items()}
            codedict_neutral['exp'] = torch.zeros_like(codedict_neutral['exp'])
            codedict_neutral['pose'][:, 3:] = torch.zeros_like(codedict_neutral['pose'][:, 3:])
    
            if args.saveVis:
                _, visdict = deca.decode(codedict, return_vis=args.saveVis)
                cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))

                _, visdict = deca.decode(codedict_neutral, return_vis=args.saveVis)
                cv2.imwrite(os.path.join(savefolder, name + '_vis_neutral.jpg'), deca.visualize(visdict))

            ##### transform
            tform = sample['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1,2).to(device)
            original_image = sample['original_image'][None, ...].to(device)
    
            orig_opdict = deca.decode(codedict_neutral, render_orig=True, original_image=original_image, tform=tform)
            
            if args.saveVis:
                _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform, return_vis=args.saveVis)
                orig_visdict['inputs'] = original_image
                cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))

                _, orig_visdict = deca.decode(codedict_neutral, render_orig=True, original_image=original_image, tform=tform, return_vis=args.saveVis)
                orig_visdict['inputs'] = original_image
                cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size_neutral.jpg'), deca.visualize(orig_visdict))

        original_images = original_image.cpu().numpy()  # (B, C, H, W)
        predicted_landmarks = orig_opdict['landmarks2d'].cpu().numpy()  # (B, N, 2)

        del images, original_image, tform, codedict, codedict_neutral
        torch.cuda.empty_cache()
        gc.collect()

        for b in range(original_images.shape[0]):
            image = original_images[b]  # (C, H, W)
            image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]].copy()
            image = (image * 255).astype(np.uint8)

            deca_landmarks = predicted_landmarks[b]  # (68, 2)

            deca_landmarks[..., 0] = deca_landmarks[..., 0] * image.shape[1] / 2 + image.shape[1] / 2
            deca_landmarks[..., 1] = deca_landmarks[..., 1] * image.shape[0] / 2 + image.shape[0] / 2

            if anno is not None:
                key = str(int(name)) 
                if key in anno:
                    json_landmarks = np.array(anno[key]["in_the_wild"]["face_landmarks"])

                    def get_T_points(lm):
                        left_eye = lm[36:42].mean(axis=0)
                        right_eye = lm[42:48].mean(axis=0)
                        nose_tip = lm[30]
                        return np.stack([left_eye, right_eye, nose_tip])

                    T_pred = get_T_points(deca_landmarks)
                    T_gt = get_T_points(json_landmarks)

                    err_T = np.mean(np.linalg.norm(T_pred - T_gt, axis=1))

                    d_eye_gt = np.linalg.norm(T_gt[0] - T_gt[1])
                    rel_err_T = err_T / (d_eye_gt + 1e-6)

                    if rel_err_T > 0.15:  
                        print(f"[SKIP] {name}: T-shape mismatch (rel={rel_err_T:.2%})", flush=True)
                        continue

            aligned_img = recreate_aligned_images(
                name,
                os.path.join(args.inputpath, name + '.png'),
                deca_landmarks,
                output_size=args.sample_size
            )
            save_path = os.path.join(savefolder, name + '.png')
            executor.submit(aligned_img.save, save_path)

    executor.shutdown(wait=True)
    print(f'-- please check the results in {savefolder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--inputpath', default='ffhq/in-the-wild-images', type=str,
                        help='path to the test data directory')
    parser.add_argument('-j', '--jsonpath', default='ffhq/ffhq-dataset-flat.json', type=str,
                        help='path to the json annotation file')
    parser.add_argument('-s', '--savefolder', default='ffhq/realigned', type=str,
                        help='path to the output directory')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='mediapipe', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    parser.add_argument('--sample_size', default=128, type=int,
                        help='output size for aligned/cropped images')
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--saveVis', action='store_true', help='whether to save visualization of output')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of threads for saving images')
    main(parser.parse_args())