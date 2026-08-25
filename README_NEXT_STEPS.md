# DetectLite next-stage test pipeline

Adds evaluation/profiling utilities to the existing DetectLite project.
It does not include VOC data or checkpoints.

## Run
1. Copy these files into the existing project.
2. Keep the existing MobileViT-XS backbone/FPN/head.
3. Run model profiling.
4. Run threshold diagnostics.
5. Run VOC mAP evaluation.
6. Only then change training/architecture and compare results.

Example:

python -m evaluation.profile_model

python -m scripts.threshold_sweep \
  --data datasets/VOCdevkit/VOC2012 \
  --checkpoint checkpoints/mobilevit_xs_v2_3epoch.pt

python -m evaluation.voc_map \
  --data datasets/VOCdevkit/VOC2012 \
  --checkpoint checkpoints/mobilevit_xs_v2_3epoch.pt \
  --split val \
  --score-threshold 0.05 \
  --iou-threshold 0.5
