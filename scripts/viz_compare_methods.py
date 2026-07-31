"""5-sample × 6-column strip:
   RGB | LiDAR GT (sparse) | 4 model predictions
     - shape_lidar               (dense_gt = depth_refined_grid)
     - shape_lidar_edge_res      (dense_gt = depth_edge_res)
     - shape_lidar_grad_shape          (refined_grid)
     - shape_lidar_grad_shape_edge_res (edge_res)
No titles, tight gaps."""
import os, sys, random, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose

sys.path.insert(0, os.path.abspath("."))
from src.util.loaders import build_loaders
from scripts.poisson_dense_gt_prototype import denorm_rgb

# eval.py has the model-build helper we need
spec = importlib.util.spec_from_file_location("eval_mod",
                                              "scripts/eval.py")
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)

torch.manual_seed(0); np.random.seed(0); random.seed(0)
N_SAMPLES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = [
    ("shape_lidar",                     "/workspace/RadarTaco/output/shape_lidar/best.pt"),
    ("shape_lidar_edge_res",            "/workspace/RadarTaco/output/shape_lidar_edge_res/best.pt"),
    ("shape_lidar_grad_shape",          "/workspace/RadarTaco/output/shape_lidar_grad_shape/best.pt"),
    ("shape_lidar_grad_shape_edge_res", "/workspace/RadarTaco/output/shape_lidar_grad_shape_edge_res/best.pt"),
]

with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
    cfg_base = compose(config_name="config", overrides=[
        "+experiment=shape_lidar_grad_cos",
        "dataset.dense_gt_dir=depth_refined",
    ])
train_loader, _, _ = build_loaders(cfg_base)
_ds = train_loader.dataset
if hasattr(_ds, "dataset"): _ds = _ds.dataset
_ds.augmentation = False
_ds.photometric_aug = None
batch = next(iter(train_loader))
H, W = batch["depth_gt_dense"].shape[-2:]

# Run inference for each checkpoint on the same 5 samples
preds = {}
for name, ckpt_path in CHECKPOINTS:
    print(f"loading {name} ← {ckpt_path}")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg_base, resolve=True))
    cfg.checkpoint = ckpt_path
    eval_mod._maybe_override_model_from_ckpt(cfg)
    model = eval_mod._build_eval_model(cfg, DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    with torch.inference_mode():
        rgb_in    = batch["rgb_norm"][:N_SAMPLES].to(DEVICE)
        radar_pts = batch["radar_points"][:N_SAMPLES].to(DEVICE)
        radar_msk = batch["radar_mask"][:N_SAMPLES].to(DEVICE)
        pred = model(rgb_in, radar_pts, radar_msk, None)  # independent mode
    preds[name] = pred[:, 0].float().cpu().numpy()
    del model, ckpt
    torch.cuda.empty_cache()

vmin, vmax, cmap = 0, 80, "viridis"
FIG_W = 36
COL_W = FIG_W / 6
ROW_H = COL_W * (H / W)
fig, axes = plt.subplots(
    N_SAMPLES, 6, figsize=(FIG_W, ROW_H * N_SAMPLES),
    gridspec_kw={"wspace": 0.02, "hspace": 0.005},
)

for s in range(N_SAMPLES):
    lidar  = batch["depth_gt_lidar"][s, 0].numpy()
    mask_l = batch["valid_mask_lidar"][s, 0].numpy().astype(bool)
    mask_d = batch["valid_mask_dense"][s, 0].numpy().astype(bool)
    rgb    = batch["rgb_norm"][s].numpy()

    axes[s, 0].imshow(denorm_rgb(rgb))
    axes[s, 0].axis("off")

    axes[s, 1].imshow(np.zeros((H, W, 3)), extent=(0, W, H, 0))
    ys_l, xs_l = np.where(mask_l & mask_d)
    axes[s, 1].scatter(xs_l, ys_l, c=lidar[ys_l, xs_l],
                       vmin=vmin, vmax=vmax, cmap=cmap, s=5, marker="s")
    axes[s, 1].set_xlim(0, W); axes[s, 1].set_ylim(H, 0)
    axes[s, 1].axis("off")

    for col, (name, _) in enumerate(CHECKPOINTS, start=2):
        axes[s, col].imshow(preds[name][s], vmin=vmin, vmax=vmax, cmap=cmap)
        axes[s, col].axis("off")

out = "/workspace/RadarTaco/output/pred_compare_5x6.png"
plt.savefig(out, dpi=80, bbox_inches="tight", pad_inches=0.05)
plt.close()
print(f"saved → {out}")
