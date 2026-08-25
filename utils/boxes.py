import torch
from torchvision.ops import batched_nms


def decode_predictions(
    outputs,
    score_threshold=0.05,
    nms_iou=0.5,
    topk=300,
):
    """
    Decode anchor-free detector outputs into bounding boxes.

    The detector predicts, for every FPN location:
      - class scores
      - left/top/right/bottom box distances
      - center-ness score

    During this diagnostic stage, classification probability is used
    directly as the confidence score. Center-ness is not multiplied
    into the classification score because the trained model currently
    produces very low center-ness values.
    """

    strides = {
        "p3": 8,
        "p4": 16,
        "p5": 32,
    }

    batch_size = next(iter(outputs.values()))["cls"].shape[0]

    batch_boxes = [[] for _ in range(batch_size)]
    batch_scores = [[] for _ in range(batch_size)]
    batch_labels = [[] for _ in range(batch_size)]

    for level, stride in strides.items():

        pred = outputs[level]

        cls = pred["cls"].sigmoid()
        box = pred["box"]
        center = pred["center"].sigmoid()

        bsz, num_classes, h, w = cls.shape

        # Grid locations
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

        px = (xx.float() + 0.5) * stride
        py = (yy.float() + 0.5) * stride

        for b in range(bsz):

            # -------------------------------------------------
            # Classification confidence
            # -------------------------------------------------
            #
            # Diagnostic version:
            # confidence = class probability
            #
            # We are intentionally NOT doing:
            #
            # cls * center
            #
            # because the center predictions are currently
            # extremely small.
            # -------------------------------------------------

            scores, labels = (
                cls[b]
                .reshape(num_classes, -1)
                .max(dim=0)
            )

            keep = scores > score_threshold

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

            # -------------------------------------------------
            # Select locations above confidence threshold
            # -------------------------------------------------

            flat_indices = torch.arange(
                h * w,
                device=cls.device,
            )[keep]

            iy = flat_indices // w
            ix = flat_indices % w

            cx = px[iy, ix]
            cy = py[iy, ix]

            # -------------------------------------------------
            # Predicted box distances
            # -------------------------------------------------

            distances = (
                box[b]
                .permute(1, 2, 0)
                .reshape(-1, 4)[keep]
            )

            left = distances[:, 0]
            top = distances[:, 1]
            right = distances[:, 2]
            bottom = distances[:, 3]

            # -------------------------------------------------
            # Convert distances to xyxy boxes
            # -------------------------------------------------

            decoded = torch.stack(
                [
                    cx - left,
                    cy - top,
                    cx + right,
                    cy + bottom,
                ],
                dim=1,
            )

            decoded = decoded.clamp(
                min=0,
                max=224,
            )

            current_scores = scores[keep]
            current_labels = labels[keep]

            # -------------------------------------------------
            # Keep only top-k predictions per feature level
            # -------------------------------------------------

            if current_scores.numel() > topk:

                indices = (
                    current_scores
                    .topk(topk)
                    .indices
                )

                decoded = decoded[indices]

                current_scores = (
                    current_scores[indices]
                )

                current_labels = (
                    current_labels[indices]
                )

            batch_boxes[b].append(decoded)
            batch_scores[b].append(current_scores)
            batch_labels[b].append(current_labels)

    # ---------------------------------------------------------
    # Combine all FPN levels for every image
    # ---------------------------------------------------------

    final_boxes = []
    final_scores = []
    final_labels = []

    for b in range(batch_size):

        if len(batch_boxes[b]) == 0:

            final_boxes.append(
                torch.empty(
                    (0, 4),
                    device=next(iter(outputs.values()))[
                        "cls"
                    ].device,
                )
            )

            final_scores.append(
                torch.empty(
                    (0,),
                    device=next(iter(outputs.values()))[
                        "cls"
                    ].device,
                )
            )

            final_labels.append(
                torch.empty(
                    (0,),
                    dtype=torch.long,
                    device=next(iter(outputs.values()))[
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

        # -----------------------------------------------------
        # Class-aware NMS
        # -----------------------------------------------------

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

        final_boxes.append(boxes)
        final_scores.append(scores)
        final_labels.append(labels)

    return (
        final_boxes,
        final_scores,
        final_labels,
    )
