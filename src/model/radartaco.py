"""Top-level RadarTaco model (paper §3.1) — independent inference mode."""
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .image_encoder import ImageEncoder
from .radar_encoder import build_radar_encoder
from .radar_fusion import PyramidRadarFusion
from .depth_decoder import DepthDecoder
from .aux_branch import AuxiliaryDepthBranch


class RadarTaco(nn.Module):
    """ResNet18 + (MLP|GNN) radar encoder + Pyramid Radar-centered fusion + U-Net decoder.

    Optional plug-in mode (paper §3.3 + Supp §8.3): set `use_aux_branch=True`
    to add an auxiliary input branch that fuses an initial relative depth
    map `D*` with image features at every pyramid scale. When `use_aux_branch`
    is True, the model's `forward` accepts an extra `rel_depth` argument.
    During training, half the samples are typically given real `D*` (from a
    frozen mono depth predictor like DPT-Hybrid) and the other half are
    given zeros — this is paper §3.4's "two equal portions" recipe and lets
    a single model serve both independent and plug-in inference modes.
    """

    def __init__(
        self,
        radar_encoder_name: str = "gnn",
        max_depth: float = 100.0,
        max_radar_points: int = 128,
        k_neighbors: int = 20,
        a_l: Tuple[float, float, float] = (48.0, 32.0, 16.0),
        radar_channels: Tuple[int, int, int] = (64, 128, 512),
        attn_heads: int = 4,
        mlp_hidden: int = 128,
        pretrained_image_encoder: bool = True,
        output_mode: str = "metric",                  # 'metric' or 'inverse'
        min_depth_clip: float = 0.5,                  # used by 'inverse' mode
        multi_scale: bool = False,                    # deep supervision aux heads
        multi_scale_levels: Tuple[int, ...] = (2, 4, 8, 16),
        use_aux_branch: bool = False,                 # paper §3.3 plug-in branch
        # Depth-specialised MoE fusion. `moe_at_l=(2,)` enables bottleneck
        # MoE (both node_blocks[2] and edge_blocks[2]) — validated in our
        # depth-specialist experiments. `moe_stage`: 1 → teacher-forcing
        # using a per-sample bin index derived from `depth_gt_dense`; 2 →
        # router self-decides. `moe_bins` gives the depth-range bin edges
        # used to compute the teacher label.
        moe_at_l: Optional[Sequence[int]] = None,
        moe_n_experts: int = 3,
        moe_use_shared: bool = True,
        moe_top_k: Optional[int] = None,
        moe_stage: int = 2,
        moe_bins: Sequence[float] = (0.0, 20.0, 50.0, 100.0),
        moe_router_arch: str = "conv1x1",
        moe_expert_ch_ratio: float = 1.0,
        moe_router_gt_type: str = "hard",
        moe_overlap_bins: Optional[Sequence[Sequence[float]]] = None,
        moe_shared_gate_mode: str = "always_on",
        moe_shared_aux: bool = False,
        moe_per_spec_aux: bool = False,
        # DPT-style multi-scale router options — used only when
        # `moe_router_arch == "dpt"`. See DPTRouter for details.
        moe_router_dpt_source_layers: Optional[Sequence[int]] = None,
        moe_router_dpt_fusion_ch: int = 128,
        # Two-stage pre-fusion MoE options.
        moe_pre_fusion_enabled: bool = False,
        moe_pre_fusion_feed_experts: bool = False,
        moe_pre_fusion_ch_ratio: float = 1.0,
        moe_pre_fusion_router_detach: bool = False,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(pretrained=pretrained_image_encoder)
        self.use_aux_branch = bool(use_aux_branch)
        if self.use_aux_branch:
            self.aux_branch = AuxiliaryDepthBranch(self.image_encoder.feat_channels)
        self.radar_encoder = build_radar_encoder(
            name=radar_encoder_name,
            channels=radar_channels,
            k_neighbors=k_neighbors,
            mlp_hidden=mlp_hidden,
        )
        self.moe_at_l = tuple(moe_at_l) if moe_at_l else ()
        self.moe_stage = int(moe_stage)
        self.moe_bins = tuple(float(x) for x in moe_bins)
        self.moe_router_gt_type = str(moe_router_gt_type)
        assert len(self.moe_bins) >= 2, "moe_bins needs at least [lo, hi]"
        self.moe_overlap_bins = (
            tuple(tuple(float(x) for x in pair) for pair in moe_overlap_bins)
            if moe_overlap_bins is not None else None
        )
        self.moe_shared_gate_mode = str(moe_shared_gate_mode)
        # Enable auxiliary shared-only depth head during training. When True
        # and MoE is on, `forward` also produces `depth_shared` — the depth
        # predicted from the shared expert alone (no specialists). The loss
        # can then supervise it directly (see `w_shared_aux` in ComposedLoss)
        # to train the shared expert as a standalone generalist predictor,
        # rather than as a residual correction on top of specialists.
        self.moe_shared_aux = bool(moe_shared_aux)
        # Enable per-specialist "what-if" depth heads during training. When
        # True and MoE is on, `forward` also produces `depth_per_spec`
        # (list of K depth predictions, each with the gate forced one-hot
        # on one specialist). Consumed by ComposedLoss with `w_task_router`
        # to compute a task-aware router loss: router probs are pushed to
        # concentrate on whichever specialist would produce the LOWEST task
        # loss on each token, rather than just matching pixel-bin majority.
        # Adds K extra decoder passes per training step (K = n_experts).
        self.moe_per_spec_aux = bool(moe_per_spec_aux)
        # cache experts count for the forward loop below
        self._moe_n_experts = int(moe_n_experts)
        self.radar_fusion = PyramidRadarFusion(
            radar_channels=radar_channels,
            img_channels=self.image_encoder.feat_channels,
            a_l=a_l,
            max_radar_points=max_radar_points,
            heads=attn_heads,
            moe_at_l=self.moe_at_l,
            moe_n_experts=int(moe_n_experts),
            moe_use_shared=bool(moe_use_shared),
            moe_top_k=moe_top_k,
            moe_bins=self.moe_bins,
            moe_router_arch=str(moe_router_arch),
            moe_expert_ch_ratio=float(moe_expert_ch_ratio),
            moe_router_gt_type=self.moe_router_gt_type,
            moe_overlap_bins=self.moe_overlap_bins,
            moe_shared_gate_mode=self.moe_shared_gate_mode,
            moe_router_dpt_source_layers=(
                tuple(moe_router_dpt_source_layers)
                if moe_router_dpt_source_layers is not None else None),
            moe_router_dpt_fusion_ch=int(moe_router_dpt_fusion_ch),
            moe_pre_fusion_enabled=bool(moe_pre_fusion_enabled),
            moe_pre_fusion_feed_experts=bool(moe_pre_fusion_feed_experts),
            moe_pre_fusion_ch_ratio=float(moe_pre_fusion_ch_ratio),
            moe_pre_fusion_router_detach=bool(moe_pre_fusion_router_detach),
        )
        self.depth_decoder = DepthDecoder(
            feat_channels=self.image_encoder.feat_channels,
            max_depth=max_depth,
            output_mode=output_mode,
            min_depth_clip=min_depth_clip,
            multi_scale=multi_scale,
            multi_scale_levels=tuple(multi_scale_levels),
        )

    @staticmethod
    def _compute_router_gt(depth_gt_dense: torch.Tensor,
                           bins: Sequence[float]) -> torch.Tensor:
        """[DEPRECATED — per-sample argmax] left for reference / diagnostics.

        Per-token GT for the MoE routing is now computed inside each
        `MoEFusionBlock.forward` (downsampled to that block's spatial
        resolution and bucketized). See `MoEFusionBlock._depth_to_bin`.
        """
        d = depth_gt_dense
        K = len(bins) - 1
        counts = []
        for i in range(K):
            lo, hi = bins[i], bins[i + 1]
            counts.append(((d >= lo) & (d < hi)).sum(dim=(1, 2, 3)))
        counts = torch.stack(counts, dim=-1)
        return counts.argmax(dim=-1)

    def forward(
        self,
        rgb: torch.Tensor,             # (B, 3, H, W) in [-1, 1]
        radar_points: torch.Tensor,    # (B, K, 3) image-space (x_pix, y_pix, depth)
        radar_mask: torch.Tensor,      # (B, K) bool
        rel_depth: torch.Tensor = None,  # (B, 1, H, W) optional initial relative depth
        depth_gt_dense: Optional[torch.Tensor] = None,   # (B, 1, H, W) — used
                                          # to compute per-sample MoE router GT
                                          # during stage-1 training. Ignored at
                                          # eval time or when MoE is off.
    ):
        image_w = rgb.shape[-1]
        feats = self.image_encoder(rgb)
        if self.use_aux_branch:
            # Independent mode is realized by passing zeros — paper §3.4
            # trains both modes by stochastically zeroing rel_depth.
            if rel_depth is None:
                rel_depth = torch.zeros(rgb.shape[0], 1, rgb.shape[2], rgb.shape[3],
                                        device=rgb.device, dtype=rgb.dtype)
            feats = self.aux_branch(rel_depth, feats)
        N_list, E_list = self.radar_encoder(radar_points, radar_mask)

        # Per-token router_gt (majority-vote of GT bins) is computed inside
        # each MoE block whenever `depth_gt_dense` is passed. Two flags
        # control its use:
        #   • gt_source: passed only during TRAINING with MoE + GT present.
        #     Ensures router_gts is emitted (needed for CE loss) in BOTH
        #     stage 1 and stage 2 training.
        #   • teacher_force: True (stage 1) → gate = one_hot(GT). False
        #     (stage 2) → self-routing; router_gt still returned for CE.
        # Result: stage-2 with w_router>0 now trains router via CE anchor,
        # solving the "router frozen after stage 1" issue caused by top_k=1
        # blocking task-loss gradient.
        # `_oracle_eval` is an ATTRIBUTE (set on the module from outside, e.g.
        # by the trainer's oracle-val pass) — treated like training for both
        # gt_source (needs GT to compute router_gt) and teacher_force (drive
        # gate from GT instead of router logits).
        oracle_eval = bool(getattr(self, "_oracle_eval", False))
        gt_source = (depth_gt_dense
                     if (self.moe_at_l and (self.training or oracle_eval)
                         and depth_gt_dense is not None)
                     else None)
        teacher_force = (self.moe_stage == 1 and self.training) or oracle_eval

        # Auxiliary shared-only path: only during training and only when
        # explicitly enabled + MoE is on. Adds a second depth-decoder pass
        # on the shared-only fused features. Eval never runs it.
        want_aux = (self.moe_shared_aux and self.training
                    and bool(self.moe_at_l))
        if want_aux:
            fused, router_logits, router_gts, fused_shared = self.radar_fusion(
                feats, N_list, E_list,
                radar_points, radar_mask, image_w=image_w,
                depth_gt_dense=gt_source,
                teacher_force=teacher_force,
                return_shared_only=True,
                rel_depth=rel_depth,
            )
        else:
            fused, router_logits, router_gts = self.radar_fusion(
                feats, N_list, E_list,
                radar_points, radar_mask, image_w=image_w,
                depth_gt_dense=gt_source,
                teacher_force=teacher_force,
                rel_depth=rel_depth,
            )
            fused_shared = None
        depth = self.depth_decoder(fused, out_hw=rgb.shape[-2:])
        depth_shared = (self.depth_decoder(fused_shared, out_hw=rgb.shape[-2:])
                        if fused_shared is not None else None)

        # Per-specialist depth predictions for task-aware router loss.
        # Only during training, only in stage 2 (self-routing), with MoE
        # enabled. Skipped in stage 1 because:
        #   (a) gate is teacher-forced there, so router is trained
        #       independently as a bin classifier — task_router loss
        #       would fight that direct supervision.
        #   (b) specialists are still early in training; per-spec errors
        #       are dominated by initialization noise, not real
        #       specialization → unreliable target for the router.
        # Reuses cached `fused` for non-MoE levels; only re-runs the
        # MoE-level blocks with the gate forced one-hot on each specialist
        # (shared skipped). K extra decoder passes.
        depth_per_spec: Optional[list] = None
        if (self.moe_per_spec_aux and self.training and bool(self.moe_at_l)
                and self.moe_stage == 2):
            # Compute per-spec forwards inside no_grad. `depth_per_spec` is
            # consumed ONLY for the task-aware router LOSS TARGET (also
            # wrapped in no_grad in the loss), so gradient never flows back
            # through these K decoder passes — saving activation memory
            # equal to K decoder forwards worth.
            depth_per_spec = []
            with torch.no_grad():
                for k in range(self._moe_n_experts):
                    fused_k = self.radar_fusion.forward_per_spec(
                        feats, N_list, E_list,
                        radar_points, radar_mask, image_w=image_w,
                        spec_idx=k, main_out=fused)
                    depth_k = self.depth_decoder(fused_k, out_hw=rgb.shape[-2:])
                    # Take top-res tensor if multi-scale dict returned.
                    depth_per_spec.append(depth_k["depth"] if isinstance(depth_k, dict)
                                           else depth_k)

        # If no MoE, keep legacy return type (Tensor or existing dict).
        if not router_logits:
            return depth

        # Otherwise, always return dict — extend the (possibly multi-scale)
        # depth output with router bookkeeping. `router_gts` matches
        # `router_logits` positionally; each entry is (B, H_block, W_block)
        # int64 in stage-1 or None in stage-2/eval.
        result = depth if isinstance(depth, dict) else {"depth": depth}
        result["router_logits"] = router_logits
        # Emit router_gt_type so the loss can pick softmax-CE (hard/soft) or
        # sigmoid-BCE (overlap) per bin. Included on every forward — loss
        # only reads it when router_gts is non-None.
        result["router_gt_type"] = self.moe_router_gt_type
        result["router_gts"] = router_gts
        # Emit moe_bins so the loss can bucketize `depth` for self-distillation
        # (target = pixel_fraction of the model's own final depth prediction).
        result["moe_bins"] = self.moe_bins
        if depth_shared is not None:
            # Shared-only depth for the auxiliary task loss. If the main
            # decoder returned a dict (multi-scale), only the top-res depth
            # tensor is passed for aux — aux never supervises multi-scale.
            result["depth_shared"] = (
                depth_shared["depth"] if isinstance(depth_shared, dict)
                else depth_shared)
        if depth_per_spec is not None:
            # List of K depth tensors, each (B, 1, H, W) — per-specialist
            # "what-if" prediction with gate forced to that spec. Consumed
            # by the task-aware router loss.
            result["depth_per_spec"] = depth_per_spec
        return result
