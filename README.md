# RadarTaco

TacoDepth 논문(arXiv:2504.11773) 기반의 효율적인 Radar-Camera depth estimation 모델입니다. ResNet18 image encoder + (MLP/GNN swappable) radar encoder + paper §3.2 pyramid radar-centered attention fusion 으로 구성된 single-stage independent inference 모델로, RadarMarigold의 무거운 SD 백본을 대체합니다.

## Focus 영역

1. **Radar-Camera Fusion** — paper §3.2 Pyramid + Radar-centered Flash Attention
2. **Simulation Dataset** — Hypersim/vKITTI2 활용 (pretrain→finetune AND mixed batch + sim-only aux loss)
3. **Long Range** — `0-50/0-70/0-80m` overall + `50-80m` far-only 별도 평가
4. **Dark Scenario** — nuScenes scene description 기반 day/night split 자동 생성 + 별도 metrics

## 디렉토리

```
config/        Hydra config (model / dataset / training / loss / experiment)
src/           라이브러리 코드 (model, dataset, loss, evaluation, trainer, util)
scripts/      실행 entry-point (train, eval, make_splits, inspect_dataset)
tests/        smoke test
```

## 설치

```bash
cd /workspace/RadarTaco
pip install -r requirements.txt
```

## 데이터 (이미 전처리됨)

- nuScenes: `/data/public/nuScenes/derived/{radar, depth_lidar, depth_interp, depth_acc, splits}/`
- Hypersim: `/data/public/Hypersim/{train,val,test}/ai_*/`
- vKITTI2: `/data/public/vkitti2/{rgb,depth}/Scene*/...`

## 1회성 셋업

```bash
# nuScenes day/night split 생성 (val_day.txt / val_night.txt)
python scripts/make_splits.py
```

## 학습

```bash
# Baseline (paper-faithful, GNN encoder, nuScenes only)
python scripts/train.py +experiment=baseline_gnn

# MLP encoder ablation
python scripts/train.py +experiment=baseline_mlp

# Mixed batch + sim-only edge/gradient aux loss
python scripts/train.py +experiment=mixed_with_aux

# Sim pretrain → nuScenes finetune (2-stage)
python scripts/train.py +experiment=sim_pretrain_finetune training.stage=sim_pretrain
python scripts/train.py +experiment=sim_pretrain_finetune training.stage=nuscenes_finetune \
    training.resume_from=output/sim_pretrain/.../best.pt
```

## 평가

```bash
python scripts/eval.py checkpoint=output/.../best.pt
```

평가 결과는 `metrics.json` 에 5개 카테고리로 출력됩니다: `overall`, `per_range` (10m bins), `far` (50-80m), `day`, `night`.

## 참고 (수정하지 않음)

- `/workspace/TacoDepth/` — 임시 paper-기반 구현 (구조 참고용)
- `/workspace/RadarMarigold/` — SD 백본 prior work
- `/workspace/RGM/`, `/workspace/dgcnn/` — GNN primitives
