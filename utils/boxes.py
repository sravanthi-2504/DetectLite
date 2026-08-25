import torch
from torchvision.ops import batched_nms


def decode_predictions(
    outputs,
    score_threshold=0.05,
    nms_iou=0.5,
    topk=300,
):
    """
    Decode V3 anchor-free predictions.

    The model predicts l/t/r/b distances normalized by
    the corresponding FPN stride.

    Therefore:

        pixel_distance = predicted_distance * stride

    Confidence uses:

        class_probability * centerness

    This matches the V3 training formulation.
    """

    strides = {
        "p3": 8,
        "p4": 16,
        "p5": 32,
    }

    batch_size = next(
        iter(outputs.values())
    )["cls"].shape[0]

    batch_boxes = [
        [] for _ in range(batch_size)
    ]

    batch_scores = [
        [] for _ in range(batch_size)
    ]

    batch_labels = [
        [] for _ in range(batch_size)
    ]

    for level, stride in strides.items():

        pred = outputs[level]

        cls = pred["cls"].sigmoid()

        center = (
            pred["center"].sigmoid()
        )

        box = pred["box"]

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

        cx = (
            xx.float() + 0.5
        ) * stride

        cy = (
            yy.float() + 0.5
        ) * stride

        for b in range(bsz):

            # --------------------------------------------------
            # Class × centerness confidence
            # --------------------------------------------------

            combined = (
                cls[b]
                *
                center[b]
            )

            scores, labels = (
                combined
                .reshape(
                    num_classes,
                    -1,
                )
                .max(dim=0)
            )

            keep = (
                scores
                >= score_threshold
            )

            if not keep.any():

                batch_boxes[b].append(
                    torch.empty(
                        (0, 4),
                        device=cls.device,
                    )
                )

                batch_scores[b].append(
                    torch.empty(
                        (0,),
                        device=cls.device,
                    )
                )

                batch_labels[b].append(
                    torch.empty(
                        (0,),
                        dtype=torch.long,
                        device=cls.device,
                    )
                )

                continue

            flat_indices = (
                torch.arange(
                    h * w,
                    device=cls.device,
                )[keep]
            )

            iy = (
                flat_indices // w
            )

            ix = (
                flat_indices % w
            )

            point_x = (
                cx[iy, ix]
            )

            point_y = (
                cy[iy, ix]
            )

            # --------------------------------------------------
            # Normalized distances → pixel distances
            # --------------------------------------------------

            distances = (
                box[b]
                .permute(
                    1,
                    2,
                    0,
                )
                .reshape(
                    -1,
                    4,
                )[keep]
            )

            distances = (
                distances * stride
            )

            left = distances[:, 0]
            top = distances[:, 1]
            right = distances[:, 2]
            bottom = distances[:, 3]

            # --------------------------------------------------
            # Decode xyxy
            # --------------------------------------------------

            decoded = torch.stack(
                [
                    point_x - left,
                    point_y - top,
                    point_x + right,
                    point_y + bottom,
                ],
                dim=1,
            )

            decoded = decoded.clamp(
                min=0.0,
                max=224.0,
            )

            current_scores = (
                scores[keep]
            )

            current_labels = (
                labels[keep]
            )

            # --------------------------------------------------
            # Per-level top-k
            # --------------------------------------------------

            if (
                current_scores.numel()
                > topk
            ):

                indices = (
                    current_scores
                    .topk(topk)
                    .indices
                )

                decoded = (
                    decoded[indices]
                )

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

    # ----------------------------------------------------------
    # Combine FPN levels
    # ----------------------------------------------------------

    final_boxes = []
    final_scores = []
    final_labels = []

    for b in range(batch_size):

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

        # ------------------------------------------------------
        # Class-aware NMS
        # ------------------------------------------------------

        if boxes.numel() > 0:

            keep = batched_nms(
                boxes,
                scores,
                labels,
                nms_iou,
            )

            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        final_boxes.append(
            boxes
        )

        final_scores.append(
            scores
        )

        final_labels.append(
            labels
        )

    return (
        final_boxes,
        final_scores,
        final_labels,
    )
