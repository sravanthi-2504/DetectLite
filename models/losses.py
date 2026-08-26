import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits,
    targets,
    alpha=0.25,
    gamma=2.0,
):
    """
    Sigmoid focal loss.

    Returns unreduced loss so the caller can normalize
    using the number of positive locations.
    """

    prob = logits.sigmoid()

    ce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    p_t = (
        prob * targets
        + (1.0 - prob) * (1.0 - targets)
    )

    focal_weight = (1.0 - p_t) ** gamma

    alpha_t = (
        alpha * targets
        + (1.0 - alpha) * (1.0 - targets)
    )

    return ce_loss * focal_weight * alpha_t


def giou_loss(
    pred_dist,
    target_dist,
    px,
    py,
):
    """
    Generalized IoU loss for anchor-free l/t/r/b distances.

    pred_dist and target_dist are in normalized feature-map
    coordinates.

    px, py are the feature-map coordinates of the positive
    locations.
    """

    pred_l = pred_dist[:, 0]
    pred_t = pred_dist[:, 1]
    pred_r = pred_dist[:, 2]
    pred_b = pred_dist[:, 3]

    tgt_l = target_dist[:, 0]
    tgt_t = target_dist[:, 1]
    tgt_r = target_dist[:, 2]
    tgt_b = target_dist[:, 3]

    # Predicted boxes.
    pred_x1 = px - pred_l
    pred_y1 = py - pred_t
    pred_x2 = px + pred_r
    pred_y2 = py + pred_b

    # Target boxes.
    tgt_x1 = px - tgt_l
    tgt_y1 = py - tgt_t
    tgt_x2 = px + tgt_r
    tgt_y2 = py + tgt_b

    # Intersection.
    inter_x1 = torch.maximum(
        pred_x1,
        tgt_x1,
    )

    inter_y1 = torch.maximum(
        pred_y1,
        tgt_y1,
    )

    inter_x2 = torch.minimum(
        pred_x2,
        tgt_x2,
    )

    inter_y2 = torch.minimum(
        pred_y2,
        tgt_y2,
    )

    inter_w = (
        inter_x2 - inter_x1
    ).clamp(min=0)

    inter_h = (
        inter_y2 - inter_y1
    ).clamp(min=0)

    intersection = inter_w * inter_h

    # Areas.
    pred_w = (
        pred_x2 - pred_x1
    ).clamp(min=0)

    pred_h = (
        pred_y2 - pred_y1
    ).clamp(min=0)

    tgt_w = (
        tgt_x2 - tgt_x1
    ).clamp(min=0)

    tgt_h = (
        tgt_y2 - tgt_y1
    ).clamp(min=0)

    pred_area = pred_w * pred_h
    target_area = tgt_w * tgt_h

    union = (
        pred_area
        + target_area
        - intersection
    )

    iou = (
        intersection
        / union.clamp(min=1e-8)
    )

    # Smallest enclosing box.
    enc_x1 = torch.minimum(
        pred_x1,
        tgt_x1,
    )

    enc_y1 = torch.minimum(
        pred_y1,
        tgt_y1,
    )

    enc_x2 = torch.maximum(
        pred_x2,
        tgt_x2,
    )

    enc_y2 = torch.maximum(
        pred_y2,
        tgt_y2,
    )

    enc_w = (
        enc_x2 - enc_x1
    ).clamp(min=0)

    enc_h = (
        enc_y2 - enc_y1
    ).clamp(min=0)

    enclosing_area = (
        enc_w * enc_h
    )

    giou = (
        iou
        - (
            (enclosing_area - union)
            / enclosing_area.clamp(min=1e-8)
        )
    )

    return 1.0 - giou


def smooth_l1_loss(
    pred_dist,
    target_dist,
):
    """
    Smooth-L1 regression loss for normalized
    l/t/r/b distances.
    """

    return F.smooth_l1_loss(
        pred_dist,
        target_dist,
        reduction="none",
    ).mean(dim=1)


def detection_loss(
    outputs,
    targets,
    image_size=224,
):
    """
    V5 anchor-free detection loss.

    V5 keeps the V4 assignment strategy but changes
    the box regression objective.

    Components:

        classification:
            sigmoid focal loss

        box regression:
            75% Smooth-L1
            25% GIoU

        centerness:
            binary cross entropy

    The regression loss is computed only at positive
    locations.
    """

    device = next(
        iter(outputs.values())
    )["cls"].device

    strides = {
        "p3": 8,
        "p4": 16,
        "p5": 32,
    }

    total_cls = torch.tensor(
        0.0,
        device=device,
    )

    total_box = torch.tensor(
        0.0,
        device=device,
    )

    total_ctr = torch.tensor(
        0.0,
        device=device,
    )

    total_positive = 0

    num_images = len(targets)

    if num_images == 0:
        return torch.tensor(
            0.0,
            device=device,
            requires_grad=True,
        )

    for b in range(num_images):

        boxes = targets[b]["boxes"].to(device)
        labels = targets[b]["labels"].to(device)

        for level, stride in strides.items():

            pred = outputs[level]

            _, num_classes, h, w = (
                pred["cls"].shape
            )

            cls_target = torch.zeros_like(
                pred["cls"][b]
            )

            box_target = torch.zeros_like(
                pred["box"][b]
            )

            center_target = torch.zeros_like(
                pred["center"][b]
            )

            positive_mask = torch.zeros(
                (h, w),
                dtype=torch.bool,
                device=device,
            )

            yy, xx = torch.meshgrid(
                torch.arange(
                    h,
                    device=device,
                ),
                torch.arange(
                    w,
                    device=device,
                ),
                indexing="ij",
            )

            # Feature-map locations in image pixels.
            px_pixels = (
                xx.float() + 0.5
            ) * stride

            py_pixels = (
                yy.float() + 0.5
            ) * stride

            # Feature-map normalized coordinates.
            px = xx.float() + 0.5
            py = yy.float() + 0.5

            for box, label in zip(
                boxes,
                labels,
            ):

                x1, y1, x2, y2 = box

                width = (
                    x2 - x1
                ).clamp(min=1.0)

                height = (
                    y2 - y1
                ).clamp(min=1.0)

                area = width * height

                # Same V4 scale assignment.
                if level == "p3":

                    valid_scale = (
                        area < 64.0 * 64.0
                    )

                elif level == "p4":

                    valid_scale = (
                        area >= 64.0 * 64.0
                        and
                        area < 128.0 * 128.0
                    )

                else:

                    valid_scale = (
                        area >= 128.0 * 128.0
                    )

                if not valid_scale:
                    continue

                cx = (
                    x1 + x2
                ) / 2.0

                cy = (
                    y1 + y2
                ) / 2.0

                inside = (
                    (px_pixels >= x1)
                    &
                    (px_pixels <= x2)
                    &
                    (py_pixels >= y1)
                    &
                    (py_pixels <= y2)
                )

                if not inside.any():
                    continue

                # Select the feature location closest
                # to the object's center.
                distance_to_center = (
                    (px_pixels - cx) ** 2
                    +
                    (py_pixels - cy) ** 2
                )

                distance_to_center = (
                    distance_to_center.masked_fill(
                        ~inside,
                        float("inf"),
                    )
                )

                flat_index = (
                    distance_to_center.argmin()
                )

                iy = (
                    flat_index // w
                )

                ix = (
                    flat_index % w
                )

                # Pixel distances.
                left = (
                    px_pixels[iy, ix] - x1
                )

                top = (
                    py_pixels[iy, ix] - y1
                )

                right = (
                    x2 - px_pixels[iy, ix]
                )

                bottom = (
                    y2 - py_pixels[iy, ix]
                )

                # Normalize by feature stride.
                box_target[
                    0,
                    iy,
                    ix,
                ] = left / stride

                box_target[
                    1,
                    iy,
                    ix,
                ] = top / stride

                box_target[
                    2,
                    iy,
                    ix,
                ] = right / stride

                box_target[
                    3,
                    iy,
                    ix,
                ] = bottom / stride

                cls_target[
                    int(label),
                    iy,
                    ix,
                ] = 1.0

                # FCOS-style centerness.
                lr_min = torch.minimum(
                    left,
                    right,
                )

                lr_max = torch.maximum(
                    left,
                    right,
                )

                tb_min = torch.minimum(
                    top,
                    bottom,
                )

                tb_max = torch.maximum(
                    top,
                    bottom,
                )

                centerness = torch.sqrt(
                    (
                        lr_min
                        / lr_max.clamp(min=1e-6)
                    )
                    *
                    (
                        tb_min
                        / tb_max.clamp(min=1e-6)
                    )
                )

                center_target[
                    0,
                    iy,
                    ix,
                ] = centerness

                positive_mask[
                    iy,
                    ix,
                ] = True

            # -------------------------------------------------
            # Classification
            # -------------------------------------------------

            cls_loss = sigmoid_focal_loss(
                pred["cls"][b],
                cls_target,
            )

            num_pos = int(
                positive_mask.sum().item()
            )

            if num_pos > 0:

                total_positive += num_pos

                # -------------------------------------------------
                # Positive box regression
                # -------------------------------------------------

                pred_box = (
                    pred["box"][b]
                    .permute(1, 2, 0)
                    [positive_mask]
                )

                target_box = (
                    box_target
                    .permute(1, 2, 0)
                    [positive_mask]
                )

                positive_px = (
                    px[positive_mask]
                )

                positive_py = (
                    py[positive_mask]
                )

                giou = giou_loss(
                    pred_box,
                    target_box,
                    positive_px,
                    positive_py,
                )

                l1 = smooth_l1_loss(
                    pred_box,
                    target_box,
                )

                # V5 regression objective.
                box_loss = (
                    0.25 * giou
                    +
                    0.75 * l1
                )

                total_box += (
                    box_loss.sum()
                )

                # -------------------------------------------------
                # Centerness
                # -------------------------------------------------

                pred_center = (
                    pred["center"][b, 0]
                    [positive_mask]
                )

                target_center = (
                    center_target[0]
                    [positive_mask]
                )

                center_loss = (
                    F.binary_cross_entropy_with_logits(
                        pred_center,
                        target_center,
                        reduction="sum",
                    )
                )

                total_ctr += center_loss

            # Normalize classification by positives
            # so the enormous negative background does
            # not dominate training.
            total_cls += (
                cls_loss.sum()
                / max(num_pos, 1)
            )

    normalizer = max(
        total_positive,
        1,
    )

    # Classification is already normalized per level,
    # while box and centerness are normalized globally.
    total_cls = total_cls / (
        num_images * 3
    )

    total_box = (
        total_box
        / normalizer
    )

    total_ctr = (
        total_ctr
        / normalizer
    )

    return (
        total_cls
        +
        total_box
        +
        0.5 * total_ctr
    )
