# MobileViT Lightweight Object Detection

A research project using MobileViT-XS as a lightweight backbone, an FPN,
and an anchor-free dense detection head.

## Architecture

Image -> preprocessing -> MobileViT-XS -> FPN -> Detection Head -> NMS -> detections

The existing `models/mobilevit_backbone.py` from the project is expected to
return:

- f1: (B, 32, 112, 112)
- f2: (B, 48, 56, 56)
- f3: (B, 64, 28, 28)
- f4: (B, 80, 14, 14)
- f5: (B, 96, 7, 7)

This repository supplies the remaining detector/training/evaluation code.

## Quick test

```bash
conda activate mobilevit-det
python main.py test
python main.py profile
```

## Pascal VOC

Place VOC data so that the root contains:

```text
VOCdevkit/VOC2007/
  JPEGImages/
  Annotations/
  ImageSets/Main/train.txt
  ImageSets/Main/val.txt
```

Train:

```bash
python main.py train --data /path/to/VOCdevkit/VOC2007 --epochs 10 --batch-size 4
```

Evaluate:

```bash
python main.py evaluate --data /path/to/VOCdevkit/VOC2007 \
  --checkpoint checkpoints/baseline.pt --split val
```

Profile:

```bash
python main.py profile --checkpoint checkpoints/baseline.pt
```

Image inference:

```bash
python main.py predict --image test.jpg \
  --checkpoint checkpoints/baseline.pt
```

Webcam:

```bash
python main.py webcam --checkpoint checkpoints/baseline.pt
```

Press `q` to exit webcam mode.

## Research workflow

1. Verify the MobileViT backbone.
2. Benchmark the baseline detector.
3. Train on Pascal VOC.
4. Measure precision, recall, F1, latency, parameters and FLOPs.
5. Profile the model to identify bottlenecks.
6. Implement the proposed MobileViT efficiency improvement.
7. Repeat training and evaluation.
8. Run an ablation study.
9. Compare baseline vs optimized model.
