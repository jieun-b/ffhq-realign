import os
import cv2
import glob
import argparse
import numpy as np
import mediapipe as mp
import random
import matplotlib.pyplot as plt

mp_face = mp.solutions.face_mesh

LEFT_EYE = 33
RIGHT_EYE = 263
MOUTH = [13, 14, 78, 308, 82, 312, 87, 317] 
JAW = list(range(0, 17))                     

def extract_metrics(image, face_mesh):
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, None

    lm = results.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * w, p.y * h] for p in lm])

    # bbox
    x_min, y_min = pts[:,0].min(), pts[:,1].min()
    x_max, y_max = pts[:,0].max(), pts[:,1].max()
    bbox_area = (x_max - x_min) * (y_max - y_min)
    area_ratio = bbox_area / (w * h)

    lx, ly = pts[LEFT_EYE]
    rx, ry = pts[RIGHT_EYE]
    eye_dist = np.sqrt((lx-rx)**2 + (ly-ry)**2)
    if eye_dist < 1e-6:
        return None, None
    pts_norm = (pts - np.array([lx, ly])) / eye_dist

    return area_ratio, pts_norm

def compute_stats(files, max_imgs=5000, image_size=256, seed=42):
    if len(files) == 0:
        raise RuntimeError("No images provided")

    random.seed(seed)
    if len(files) > max_imgs:
        files = random.sample(files, max_imgs)

    area_ratios, landmarks_norm = [], []
    acc_img = np.zeros((image_size, image_size, 3), np.float64)
    count = 0

    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            area_ratio, lm_norm = extract_metrics(img, face_mesh)
            if area_ratio is None:
                continue

            area_ratios.append(area_ratio)
            landmarks_norm.append(lm_norm)

            resized = cv2.resize(img, (image_size, image_size))
            acc_img += resized.astype(np.float64)
            count += 1

    mean_face = (acc_img / max(1, count)).astype(np.uint8)
    return np.array(area_ratios), np.array(landmarks_norm), mean_face


def landmark_variance(lm_norm, region_idx):
    region = lm_norm[:, region_idx, :]
    return np.mean(np.var(region, axis=0))

def save_barplots(o_area_std, p_area_std, o_mouth_var, p_mouth_var, o_jaw_var, p_jaw_var, out_dir):
    plt.figure(figsize=(12,5))

    # Scale Consistency (bbox ratio std)
    plt.subplot(1, 2, 1)
    plt.bar(["Original FFHQ", "Realigned FFHQ"], 
            [o_area_std, p_area_std], 
            color=["#1f77b4","#ff7f0e"])
    plt.title("Scale Consistency")
    plt.ylabel("Std of BBox Area Ratio")

    # Landmark Variance (Mouth & Jaw)
    plt.subplot(1, 2, 2)
    x = np.arange(2)
    width = 0.35
    plt.bar(x - width/2, [o_mouth_var, o_jaw_var], width, 
            label="Original FFHQ", color="#1f77b4")
    plt.bar(x + width/2, [p_mouth_var, p_jaw_var], width, 
            label="Realigned FFHQ", color="#ff7f0e")
    plt.xticks(x, ["Mouth Region", "Jaw Region"])
    plt.title("Landmark Variance (Normalized Coordinates)")
    plt.ylabel("Variance")
    plt.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "metrics_barplot.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[INFO] Bar plots saved to {path}")
    
    
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    orig_files = sorted(glob.glob(os.path.join(args.original, "**", "*.png"), recursive=True) +
                        glob.glob(os.path.join(args.original, "**", "*.jpg"), recursive=True))
    prep_files = sorted(glob.glob(os.path.join(args.preprocessed, "**", "*.png"), recursive=True) +
                        glob.glob(os.path.join(args.preprocessed, "**", "*.jpg"), recursive=True))

    orig_names = {os.path.splitext(os.path.basename(f))[0]: f for f in orig_files}
    prep_names = {os.path.splitext(os.path.basename(f))[0]: f for f in prep_files}
    common_keys = sorted(list(orig_names.keys() & prep_names.keys()))

    if len(common_keys) == 0:
        raise RuntimeError("No common images found between original and preprocessed folders")

    print(f"[INFO] Found {len(common_keys)} common images")

    orig_common = [orig_names[k] for k in common_keys]
    prep_common = [prep_names[k] for k in common_keys]

    print("Processing Original FFHQ ...")
    o_area, o_lm, o_mean = compute_stats(orig_common, args.max_imgs, args.image_size, args.seed)

    print("Processing Realigned FFHQ ...")
    p_area, p_lm, p_mean = compute_stats(prep_common, args.max_imgs, args.image_size, args.seed)

    o_area_std, p_area_std = np.std(o_area), np.std(p_area)

    o_mouth_var = landmark_variance(o_lm, MOUTH)
    p_mouth_var = landmark_variance(p_lm, MOUTH)
    o_jaw_var   = landmark_variance(o_lm, JAW)
    p_jaw_var   = landmark_variance(p_lm, JAW)

    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    axes[0].imshow(cv2.cvtColor(o_mean, cv2.COLOR_RGB2BGR))
    axes[0].set_title("Original FFHQ")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(p_mean, cv2.COLOR_RGB2BGR))
    axes[1].set_title("Realigned FFHQ")
    axes[1].axis("off")

    plt.tight_layout()
    path_comp = os.path.join(args.out_dir, "meanface_comparison.png")
    plt.savefig(path_comp, dpi=200)
    plt.close()
    print(f"[INFO] Mean face comparison saved to {path_comp}")

    print("\n=== Results ===")
    print(f"Scale Consistency (bbox area ratio std):")
    print(f"  Original={o_area_std:.4f}, Preprocessed={p_area_std:.4f}")
    print(f"Landmark Variance (normalized coords):")
    print(f"  Mouth: Original={o_mouth_var:.4f}, Preprocessed={p_mouth_var:.4f}")
    print(f"  Jaw:   Original={o_jaw_var:.4f}, Preprocessed={p_jaw_var:.4f}")
    print(f"\nMean faces saved to {args.out_dir} (visual reference only).")

    save_barplots(o_area_std, p_area_std, o_mouth_var, p_mouth_var, o_jaw_var, p_jaw_var, args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True, help="Path to original dataset folder")
    parser.add_argument("--preprocessed", type=str, required=True, help="Path to preprocessed dataset folder")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--max_imgs", type=int, default=5000, help="Max number of images to process")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for mean face (visual only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
