import torch
from torchvision.ops import batched_nms


def decode_predictions(
    outputs,
    score_threshold=0.15,
    nms_iou=0.5,
    topk=300,
):
    """
    Decode anchor-free FPN predictions.

    Returns one prediction list per image in the batch.
    """

    strides = {
        "p3": 8,
        "p4": 16,
        "p5": 32,
    }

    batch_size = (
        outputs["p3"]["cls"].shape[0]
    )

    batch_boxes = [
        []
        for _ in range(batch_size)
    ]

    batch_scores = [
        []
        for _ in range(batch_size)
    ]

    batch_labels = [
        []
        for _ in range(batch_size)
    ]

    for level, stride in strides.items():

        pred = outputs[level]

        cls = pred["cls"].sigmoid()
        box = pred["box"]
        center = pred["center"].sigmoid()

        bsz, num_classes, h, w = (
            cls.shape
        )

        yy, xx = torch.meshgrid(
            torch.arange(
                h,
                device=cls.device,
            ),
            torch.arange(
                w,
                device=cls.device,
            ),
            indexing="ij",
        )

        px = (
            xx.float() + 0.5
        ) * stride

        py = (
            yy.float() + 0.5
        ) * stride

        for b in range(bsz):

            # Combine class probability and center score.
            scores_map = (
                cls[b]
                * center[b]
            )

            scores, labels = (
                scores_map
                .reshape(num_classes, -1)
                .max(dim=0)
            )

            keep = (
                scores >= score_threshold
            )

            if not keep.any():
                continue

            flat = torch.where(
                keep
            )[0]

            iy = flat // w
            ix = flat % w

            cx = px[iy, ix]
            cy = py[iy, ix]

            distances = (
                box[b]
                .permute(1, 2, 0)
                .reshape(-1, 4)[keep]
            )

            decoded = torch.stack(
                [
                    cx - distances[:, 0],
                    cy - distances[:, 1],
                    cx + distances[:, 2],
                    cy + distances[:, 3],
                ],
                dim=1,
            )

            decoded[:, 0::2] = (
                decoded[:, 0::2]
                .clamp(0, 224)
            )

            decoded[:, 1::2] = (
                decoded[:, 1::2]
                .clamp(0, 224)
            )

            current_scores = scores[
                keep
            ]

            current_labels = labels[
                keep
            ]

            if (
                current_scores.numel()
                > topk
            ):
                indices = (
                    current_scores
                    .topk(topk)
                    .indices
                )

                decoded = decoded[
                    indices
                ]

                current_scores = (
                    current_scores[
                        indices
                    ]
                )

                current_labels = (
                    current_labels[
                        indices
                    ]
                )

            batch_boxes[b].append(
                decoded
            )

            batch_scores[b].append(
                current_scores
            )

            batch_labels[b].append(
                current_labels
            )

    final_boxes = []
    final_scores = []
    final_labels = []

    for b in range(batch_size):

        if not batch_boxes[b]:

            final_boxes.append(
                torch.empty(
                    (0, 4),
                    device=outputs["p3"][
                        "cls"
                    ].device,
                )
            )

            final_scores.append(
                torch.empty(
                    0,
                    device=outputs["p3"][
                        "cls"
                    ].device,
                )
            )

            final_labels.append(
                torch.empty(
                    0,
                    dtype=torch.long,
                    device=outputs["p3"][
                        "cls"
                    ].device,
                )
            )

            continue

        boxes = torch.cat(
            batch_boxes[b],
            dim=0,
        )

        scores = torch.cat(
            batch_scores[b],
            dim=0,
        )

        labels = torch.cat(
            batch_labels[b],
            dim=0,
        )

        keep = batched_nms(
            boxes,
            scores,
            labels,
            nms_iou,
        )

        final_boxes.append(
            boxes[keep]
        )

        final_scores.append(
            scores[keep]
        )

        final_labels.append(
            labels[keep]
        )

    return (
        final_boxes,
        final_scores,
        final_labels,
    )
