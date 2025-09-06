import os
import cv2
import argparse
import numpy as np
from glob import glob
import random, math
import mediapipe as mp
import matplotlib.pyplot as plt

mp_face = mp.solutions.face_mesh

LEFT_EYE = 33
RIGHT_EYE = 263
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


def get_rotation_and_mouth(img, face_mesh):
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, None
    lm = results.multi_face_landmarks[0].landmark
    lx, ly = lm[LEFT_EYE].x * w, lm[LEFT_EYE].y * h
    rx, ry = lm[RIGHT_EYE].x * w, lm[RIGHT_EYE].y * h
    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    mx, my = lm[MOUTH_TOP].x * w, lm[MOUTH_TOP].y * h
    bx, by = lm[MOUTH_BOTTOM].x * w, lm[MOUTH_BOTTOM].y * h
    mouth_open = np.sqrt((mx - bx) ** 2 + (my - by) ** 2)
    return angle, mouth_open


def save_pair(orig_file, prep_file, tag, score, out_dir, image_size=None, show_diff=True):
    name = os.path.splitext(os.path.basename(orig_file))[0]
    img_orig = cv2.imread(orig_file)
    img_prep = cv2.imread(prep_file)

    if image_size is not None:
        img_orig = cv2.resize(img_orig, (image_size, image_size))
        img_prep = cv2.resize(img_prep, (image_size, image_size))

    panels = [img_orig, img_prep]
    if show_diff:
        diff = cv2.absdiff(img_orig, img_prep)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
        panels.append(diff)

    fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * 4, 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, img in zip(axes, panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    save_path = os.path.join(out_dir, f"{name}_{tag}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[INFO] Saved {save_path} ({tag} score={score:.2f})")


def compare_crops(original, preprocessed, out_dir, num_samples=5, subset_size=1000,
                  show_diff=True, image_size=None, seed=42):
    os.makedirs(out_dir, exist_ok=True)

    orig_files = {os.path.splitext(os.path.basename(f))[0]: f 
                  for f in glob(os.path.join(original, "*.png"))}
    prep_files = {os.path.splitext(os.path.basename(f))[0]: f 
                  for f in glob(os.path.join(preprocessed, "*.png"))}

    common_keys = sorted(set(orig_files.keys()) & set(prep_files.keys()))
    if not common_keys:
        raise ValueError("No matching files found between original and preprocessed folders!")

    random.seed(seed)
    sampled_keys = random.sample(common_keys, min(subset_size, len(common_keys)))

    diffs = []
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for key in sampled_keys:
            orig_file = orig_files[key]
            prep_file = prep_files[key]

            img_orig = cv2.imread(orig_file)
            img_prep = cv2.imread(prep_file)
            if img_orig is None or img_prep is None:
                continue

            if image_size is not None:
                img_orig = cv2.resize(img_orig, (image_size, image_size))
                img_prep = cv2.resize(img_prep, (image_size, image_size))

            ang1, mouth1 = get_rotation_and_mouth(img_orig, face_mesh)
            ang2, mouth2 = get_rotation_and_mouth(img_prep, face_mesh)
            if ang1 is None or ang2 is None:
                continue

            rot_diff = abs(ang1 - ang2)
            mouth_diff = abs(mouth1 - mouth2)
            diffs.append((rot_diff, mouth_diff, orig_file, prep_file))

    top_rot = sorted(diffs, key=lambda x: x[0], reverse=True)[:num_samples]
    top_mouth = sorted(diffs, key=lambda x: x[1], reverse=True)[:num_samples]

    for rot_diff, _, orig_file, prep_file in top_rot:
        save_pair(orig_file, prep_file, "rotation", rot_diff,
                  out_dir, image_size=image_size, show_diff=show_diff)
    for _, mouth_diff, orig_file, prep_file in top_mouth:
        save_pair(orig_file, prep_file, "mouth", mouth_diff,
                  out_dir, image_size=image_size, show_diff=show_diff)

    print(f"[INFO] Saved {len(top_rot)} rotation and {len(top_mouth)} mouth samples to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare FFHQ crop vs Realigned crop (rotation & mouth difference)")
    parser.add_argument("--original", type=str, required=True, help="Path to original dataset folder")
    parser.add_argument("--preprocessed", type=str, required=True, help="Path to realigned dataset folder")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of top samples per criterion")
    parser.add_argument("--subset_size", type=int, default=1000, help="Number of random pairs to evaluate")
    parser.add_argument("--image_size", type=int, default=256, help="Resize output images to this size (e.g., 256)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no_diff", action="store_true", help="Do not compute difference heatmap")
    args = parser.parse_args()

    compare_crops(
        args.original, args.preprocessed, args.out_dir,
        num_samples=args.num_samples,
        subset_size=args.subset_size,
        show_diff=not args.no_diff,
        image_size=args.image_size,
        seed=args.seed
    )