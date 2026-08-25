#!/usr/bin/env python3
"""Check that each of the seven radar-path fixes does at runtime what it was
meant to do, by probing a trained checkpoint on real validation frames.

Every probe is a before/after comparison. Where the "before" number can be
recovered from the same weights (fixes 2, 3, 5) the script recomputes it by
disabling the fix in place; where it cannot (fixes 1, 4, 6, 7 changed what the
network learned) it loads the pre-fix baseline checkpoint and measures both.

    python scripts/verify_radar_path_fix.py \
        --fix output/shape_lidar_grad_shape_edge_res_fix/best.pt \
        --old output/shape_lidar_grad_shape_edge_res/best.pt \
        --n 24

The MoE dispatch check needs a MoE run and is skipped otherwise:

    python scripts/verify_radar_path_fix.py \
        --fix output/radartaco_moe_stage2_v3_fix/best.pt --n 24
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.dataset.nuscenes import NuScenesRadarDepthDataset      # noqa: E402
from src.model.radar_encoder_gnn import masked_knn              # noqa: E402
from src.model.radar_fusion import RadarCenteredAttention       # noqa: E402
from src.model.radartaco import RadarTaco                       # noqa: E402

STRIDES = (2, 4, 8, 16, 32, 64)


def hr(title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")


def load_run(ckpt, device):
    cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt), "config.yaml"))
    m = RadarTaco(
        radar_encoder_name=cfg.model.radar_encoder,
        max_depth=float(cfg.dataset.max_depth),
        max_radar_points=int(cfg.dataset.max_radar_points),
        k_neighbors=int(cfg.model.k_neighbors),
        a_l=tuple(cfg.model.a_l),
        radar_channels=tuple(cfg.model.radar_channels),
        attn_heads=int(cfg.model.attn_heads),
        mlp_hidden=int(cfg.model.get("mlp_hidden", 128)),
        pretrained_image_encoder=False,
        output_mode=str(cfg.model.get("output_mode", "metric")),
        min_depth_clip=float(cfg.model.get("min_depth_clip", 0.5)),
        multi_scale=bool(cfg.model.get("multi_scale", False)),
        multi_scale_levels=tuple(cfg.model.get("multi_scale_levels", (2, 4, 8, 16))),
        use_aux_branch=bool(cfg.model.get("use_aux_branch", False)),
        moe_at_l=tuple(cfg.model.get("moe_at_l") or ()),
        moe_n_experts=int(cfg.model.get("moe_n_experts", 3)),
        moe_use_shared=bool(cfg.model.get("moe_use_shared", True)),
        moe_top_k=(int(cfg.model.get("moe_top_k"))
                   if cfg.model.get("moe_top_k") is not None else None),
        moe_stage=int(cfg.model.get("moe_stage", 2)),
        moe_bins=tuple(cfg.model.get("moe_bins", (0.0, 20.0, 50.0, 100.0))),
        moe_router_arch=str(cfg.model.get("moe_router_arch", "conv1x1")),
    )
    sd = torch.load(ckpt, map_location="cpu")["model"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    return m.to(device).eval(), cfg, missing, unexpected


def build_loader(cfg, n, split="val"):
    f = os.path.join(cfg.dataset.data_root, "splits", f"{split}.txt")
    ds = NuScenesRadarDepthDataset(
        data_root=cfg.dataset.data_root, split_file=f,
        dense_gt_dir=cfg.dataset.get("dense_gt_dir", "depth_acc"),
        lidar_gt_dir="depth_lidar",
        radar_3d_dir=cfg.dataset.get("radar_3d_dir", "radar_3d"),
        max_radar_points=int(cfg.dataset.max_radar_points),
        max_depth=float(cfg.dataset.max_depth),
        min_depth=float(cfg.dataset.min_depth),
        resize_to_hw=None, augmentation=False,
    )
    ds.samples = ds.samples[:n]
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)


# ------------------------------------------------------------------ probes --
def probe_gnn(model, batches, device, tag):
    """Fixes 1/3/4/5: edge_feat masking, k_eff clamp, kNN softmax mask, k=8."""
    enc = model.radar_encoder
    stats = {"pad_frac": [], "edge_in_pad_absmax": [], "rank1": [[], [], []],
             "row_support": [[], [], []], "nbr_dist": [], "knn_valid_frac": [],
             "bn_shift": []}

    # Capture what actually reaches the first BatchNorm inside node_gen.
    grabbed = {}
    hooks = [enc.layers[l].node_gen.conv1[0].register_forward_pre_hook(
        (lambda l: lambda mod, inp: grabbed.__setitem__(l, inp[0].detach()))(l))
        for l in range(3)]

    for b in batches:
        pts, msk = b["radar_points"].to(device), b["radar_mask"].to(device)
        with torch.no_grad():
            N_list, E_list = enc(pts, msk)
        stats["pad_frac"].append(1.0 - msk.float().mean().item())

        # (1) padded query rows must be exactly zero at the BN input
        for l in range(3):
            e = grabbed[l]                                        # (B, C, K, k)
            pad = ~msk[:, None, :, None].expand_as(e)
            stats["edge_in_pad_absmax"].append(e[pad].abs().max().item() if pad.any() else 0.0)

        # (3)/(5) kNN: are the picked neighbours all valid, and how local?
        coords = (pts[:, :, :3].transpose(1, 2) * msk[:, None, :].float())
        idx = masked_knn(coords, msk, enc.k)                       # (B,K,k)
        nbr_valid = torch.gather(msk[:, None, :].expand(-1, idx.shape[1], -1), 2, idx)
        stats["knn_valid_frac"].append(nbr_valid[msk].float().mean().item())
        ct = coords.transpose(1, 2)                                # (B,K,3)
        nbr = torch.gather(ct[:, :, None, :].expand(-1, -1, idx.shape[-1], -1), 1,
                           idx[..., None].expand(-1, -1, -1, 3))
        d = (nbr - ct[:, :, None, :]).norm(dim=-1)                 # (B,K,k)
        stats["nbr_dist"].append(d[msk].mean().item())

        # (4) is E_l still a rank-1 (query-independent) matrix?
        for l, E in enumerate(E_list):
            v = msk[0]
            sub = E[0][v][:, v].float()
            s = torch.linalg.svdvals(sub)
            stats["rank1"][l].append(((s[0] ** 2) / (s ** 2).sum()).item())
            stats["row_support"][l].append((sub > 1e-4).float().sum(-1).mean().item())

    for h in hooks:
        h.remove()

    m = lambda x: sum(x) / len(x)
    print(f"\n[{tag}]  k_neighbors = {enc.k}   패딩 비율 = {m(stats['pad_frac'])*100:.1f}%")
    print(f"  (1) node_gen BN 입력의 패딩 행 |max|      : {max(stats['edge_in_pad_absmax']):.3e}"
          f"   {'← 0이면 오염 없음' if max(stats['edge_in_pad_absmax']) < 1e-6 else '← 오염!'}")
    print(f"  (3) kNN이 고른 이웃 중 유효 비율          : {m(stats['knn_valid_frac'])*100:.2f}%")
    print(f"  (5) 이웃까지 평균 거리                    : {m(stats['nbr_dist']):.2f} m")
    for l in range(3):
        print(f"  (4) E_{l+1}  rank-1 에너지 {m(stats['rank1'][l])*100:6.2f}%"
              f"   행당 유효 후보 {m(stats['row_support'][l]):6.1f}개")
    return stats


def probe_attention(model, batches, device, tag):
    """Fixes 2/6/7: a_pix floor, null token, kv LayerNorm."""
    fuse = model.radar_fusion
    attns = []
    for l in range(fuse.L):
        for blocks in (fuse.node_blocks, fuse.edge_blocks):
            blk = blocks[l]
            inner = getattr(blk, "experts", None)
            for sub in (list(inner) if inner is not None else [blk]):
                if hasattr(sub, "attn"):
                    attns.append((l, sub.attn))

    # --- fix 2: window width per level, with and without the floor ---
    print(f"\n[{tag}]  (2) a_l 하한 — 레벨별 유효 컬럼")
    print(f"      {'lvl':<5}{'stride':>7}{'a_l':>6}{'a_pix':>7}{'컬럼(현재)':>12}"
          f"{'컬럼(하한없이)':>15}{'0-컬럼 점':>11}")
    xs = torch.cat([b["radar_points"][b["radar_mask"]][:, 3] for b in batches])
    for i, s in enumerate(STRIDES):
        a_l = attns[i][1].a_l if i < len(attns) else fuse.node_blocks[i // 2].attn.a_l
        W, sc = 1600 // s, 1.0 / s
        col = torch.arange(W).float()
        xp = xs * sc
        n_fix = ((col[None] - xp[:, None]).abs() < max(a_l * sc, 1.0)).sum(1).float()
        n_old = ((col[None] - xp[:, None]).abs() < a_l * sc).sum(1).float()
        print(f"      f{i+1:<4}{s:>7}{a_l:>6.0f}{max(a_l*sc,1.0):>7.2f}{n_fix.mean():>12.2f}"
              f"{n_old.mean():>15.2f}{(n_old==0).float().mean()*100:>10.1f}%")

    # --- fixes 6/7: null-token mass and value diversity ---
    rec = []
    for l, a in attns:
        a.record_attention = True
        a.last_attn = None
        rec.append((l, a))
    with torch.no_grad():
        b = batches[0]
        model(b["rgb_norm"].to(device), b["radar_points"].to(device), b["radar_mask"].to(device))
    null_share, val_std, kv_scale = {}, {}, {}
    for l, a in rec:
        if a.last_attn is None:
            continue
        at = a.last_attn.float()                       # (B,h,H,W,K+1)
        keep = a.last_keep                             # (B,H,W,K)
        live = keep.any(-1)                            # pixels with radar in window
        if live.any():
            nl = at[..., -1].mean(1)[live].mean().item()
            null_share.setdefault(l, []).append(nl)
        a.record_attention = False
        a.last_attn = None

    for l, a in attns:
        blk_kv = None
        for blocks in (fuse.node_blocks, fuse.edge_blocks):
            blk = blocks[l]
            for sub in (list(getattr(blk, "experts", []) or [blk])):
                if getattr(sub, "attn", None) is a:
                    blk_kv = sub
        if blk_kv is None or not hasattr(blk_kv, "norm_kv"):
            continue
        w = blk_kv.norm_kv.weight.detach()
        kv_scale.setdefault(l, []).append((w.mean().item(), w.std().item()))

    print(f"\n[{tag}]  (6) null 토큰이 흡수한 어텐션 질량 (윈도우에 레이더가 있는 픽셀 평균)")
    for l in sorted(null_share):
        v = null_share[l]
        print(f"      level {l}:  {sum(v)/len(v)*100:6.2f}%   (블록 {len(v)}개)")
    if kv_scale:
        print(f"\n[{tag}]  (7) norm_kv.weight — 1.0에서 얼마나 학습됐나")
        for l in sorted(kv_scale):
            mu = sum(x[0] for x in kv_scale[l]) / len(kv_scale[l])
            sd = sum(x[1] for x in kv_scale[l]) / len(kv_scale[l])
            print(f"      level {l}:  mean={mu:.3f}  std={sd:.3f}"
                  f"   {'← 초기값 그대로' if abs(mu-1) < 1e-3 and sd < 1e-3 else '← 학습됨'}")
    else:
        print(f"\n[{tag}]  (7) norm_kv 없음 — 수정 전 체크포인트")

    # value-vector diversity across radar points (what fix 7 targeted)
    print(f"\n[{tag}]  (7) 레이더 점별 value 벡터 분산 (edge 블록, E_l 입력)")
    with torch.no_grad():
        b = batches[0]
        pts, msk = b["radar_points"].to(device), b["radar_mask"].to(device)
        _, E_list = model.radar_encoder(pts, msk)
        for l in range(fuse.L):
            blk = fuse.edge_blocks[l]
            sub = list(getattr(blk, "experts", []) or [blk])[0]
            kv = E_list[l]
            if hasattr(sub, "norm_kv"):
                kv = sub.norm_kv(kv)
            v = sub.attn.v_proj(kv)[0][msk[0]]                    # (K_valid, C)
            inter = v.std(0).mean().item()
            print(f"      level {l}:  점간 표준편차 {inter:.4f}   ‖v‖ 평균 {v.norm(dim=-1).mean():.3f}")


def probe_kclamp(device):
    """Fix 3 in isolation: a frame with fewer valid points than k."""
    hr("(3) k_eff 클램프 — 유효 점이 k보다 적은 프레임")
    K, k = 128, 20
    for n_valid in (5, 12, 20, 40):
        coords = torch.randn(1, 3, K, device=device)
        valid = torch.zeros(1, K, dtype=torch.bool, device=device)
        valid[0, :n_valid] = True
        coords = coords * valid[:, None, :].float()
        idx = masked_knn(coords, valid, k)
        picked_valid = torch.gather(valid[:, None, :].expand(-1, K, -1), 2, idx)
        print(f"  유효 {n_valid:>3}점, k={k}  →  k_eff={idx.shape[-1]:>3}"
              f"   고른 이웃 중 유효 {picked_valid[valid].float().mean()*100:6.2f}%")


def probe_bn_padding(model, batches, device, tag):
    """Fix 1, behavioural + statistical.

    Behavioural: does the number of padded slots change the answer for the
    valid points? In train mode the BatchNorms use batch statistics, so any
    residual contamination shows up as a shift.

    Statistical: zeroing the padded rows removes the garbage *values*, but the
    zeros are still counted by BatchNorm2d, which reduces over (B, K, k). So
    compare the statistics BN actually computes against the ones it would
    compute over valid rows only."""
    enc = model.radar_encoder
    b = batches[0]
    pts, msk = b["radar_points"].to(device), b["radar_mask"].to(device)
    n = int(msk[0].sum())

    print(f"\n[{tag}]  (1) 패딩 슬롯 수를 {n+8} → {n+40} → 128로 바꿨을 때 N_3(유효 점) 변화")
    for mode in ("eval", "train"):
        enc.train(mode == "train")
        outs = []
        for K in (n + 8, n + 40, 128):
            with torch.no_grad():
                N_list, _ = enc(pts[:, :K].clone(), msk[:, :K].clone())
            outs.append(N_list[2][0][msk[0, :K]].float())
        ref = outs[0].abs().max().item()
        d1 = (outs[1] - outs[0]).abs().max().item()
        d2 = (outs[2] - outs[0]).abs().max().item()
        verdict = "← 패딩 무관" if d2 / max(ref, 1e-9) < 1e-3 else "← 패딩이 통계에 영향"
        print(f"      {mode:5s} (BN {'배치' if mode == 'train' else '러닝'} 통계)"
              f"  Δ={d1:.3e} / {d2:.3e}   스케일 {ref:7.3f}   {verdict}")
    enc.eval()

    # What BN reduces over vs. what it should reduce over.
    grabbed = {}
    hooks = [enc.layers[l].node_gen.conv1[0].register_forward_pre_hook(
        (lambda l: lambda mod, inp: grabbed.__setitem__(l, inp[0].detach().float()))(l))
        for l in range(3)]
    with torch.no_grad():
        enc(pts, msk)
    for h in hooks:
        h.remove()
    print(f"      BN이 실제로 집계하는 통계 vs 유효 행만 집계했을 때 "
          f"(패딩 {(1-msk.float().mean()).item()*100:.1f}%)")
    for l in range(3):
        e = grabbed[l]                                       # (B, C, K, k)
        v = msk[:, None, :, None].expand_as(e)
        all_m, all_v = e.mean((0, 2, 3)), e.var((0, 2, 3), unbiased=False)
        cnt = v.sum((0, 2, 3)).clamp(min=1)
        val_m = (e * v).sum((0, 2, 3)) / cnt
        val_v = ((e - val_m[None, :, None, None]) ** 2 * v).sum((0, 2, 3)) / cnt
        # Report the shift in units of the valid-only std — a ratio of means
        # is useless here because the valid-only mean passes through zero.
        sd = val_v.sqrt().clamp(min=1e-9)
        print(f"        layer {l+1}:  mean 이동 {((all_m - val_m).abs() / sd).mean():.3f} sigma"
              f"   std 비 {(all_v.sqrt() / sd).mean():.3f}"
              f"   (0.000 / 1.000이면 동일)")


def probe_moe(model, batches, device, tag):
    fuse = model.radar_fusion
    blocks = [(l, kind, b) for l in range(fuse.L)
              for kind, bl in (("node", fuse.node_blocks), ("edge", fuse.edge_blocks))
              for b in [bl[l]] if hasattr(b, "experts")]
    if not blocks:
        return
    hr(f"MoE 토큰 dispatch — 전문가별 점유율 [{tag}]")
    counts = {}
    for l, kind, blk in blocks:
        orig = blk.forward

        def wrap(f=orig, key=(l, kind)):
            def inner(feat, kv, rx, rm, iw, *a, **kw):
                out = f(feat, kv, rx, rm, iw, *a, **kw)
                return out
            return inner
        blk.forward = wrap()
    # simplest reliable measurement: hook each expert's forward and count sel
    hooks = []
    for l, kind, blk in blocks:
        for e_i, ex in enumerate(blk.experts):
            def pre(mod, args, kwargs, key=(l, kind, e_i)):
                sel = kwargs.get("sel", args[5] if len(args) > 5 else None)
                tot = args[0].shape[0] * args[0].shape[2] * args[0].shape[3]
                n = tot if sel is None else int(sel.sum())
                counts.setdefault(key, []).append(n / tot)
            hooks.append(ex.register_forward_pre_hook(pre, with_kwargs=True))
    with torch.no_grad():
        for b in batches[:4]:
            model(b["rgb_norm"].to(device), b["radar_points"].to(device),
                  b["radar_mask"].to(device))
    for h in hooks:
        h.remove()
    for l in range(fuse.L):
        for kind in ("node", "edge"):
            ks = [k for k in counts if k[0] == l and k[1] == kind]
            if not ks:
                continue
            sh = [sum(counts[k]) / len(counts[k]) * 100 for k in sorted(ks)]
            print(f"  level {l} {kind:<5}  " +
                  "  ".join(f"E{i}={v:5.1f}%" for i, v in enumerate(sh)) +
                  f"   합계 {sum(sh):5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", required=True)
    ap.add_argument("--old", default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mf, cfg, missing, unexpected = load_run(a.fix, device)
    print(f"fix ckpt: {a.fix}")
    if missing or unexpected:
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    batches = list(build_loader(cfg, a.n, a.split))
    print(f"{len(batches)} val 프레임 로드")

    hr("수정 후 (fix)")
    probe_gnn(mf, batches, device, "fix")
    probe_bn_padding(mf, batches, device, "fix")
    probe_attention(mf, batches, device, "fix")
    probe_moe(mf, batches, device, "fix")
    probe_kclamp(device)

    if a.old:
        mo, cfo, _, _ = load_run(a.old, device)
        hr("수정 전 (old baseline)")
        probe_gnn(mo, batches, device, "old")
        probe_bn_padding(mo, batches, device, "old")
        probe_attention(mo, batches, device, "old")


if __name__ == "__main__":
    main()
