import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits,
    targets,
    alpha=0.25,
    gamma=2.0,
):
    """
    Numerically stable sigmoid focal loss.

    Returns the unreduced loss so the caller can normalize
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

    focal_weight = (
        (1.0 - p_t) ** gamma
    )

    alpha_t = (
        alpha * targets
        + (1.0 - alpha) * (1.0 - targets)
    )

    return ce_loss * focal_weight * alpha_t


def box_iou_loss(
    pred_dist,
    target_dist,
    px,
    py,
):
    """
    IoU loss for anchor-free l/t/r/b distances.

    Distances are already normalized by the feature-map stride.
    Therefore both predicted and target distances are in the
    same coordinate system.

    The point is at (px, py) in normalized feature coordinates.
    """

    pred_l = pred_dist[:, 0]
    pred_t = pred_dist[:, 1]
    pred_r = pred_dist[:, 2]
    pred_b = pred_dist[:, 3]

    tgt_l = target_dist[:, 0]
    tgt_t = target_dist[:, 1]
    tgt_r = target_dist[:, 2]
    tgt_b = target_dist[:, 3]

    pred_x1 = px - pred_l
    pred_y1 = py - pred_t
    pred_x2 = px + pred_r
    pred_y2 = py + pred_b

    tgt_x1 = px - tgt_l
    tgt_y1 = py - tgt_t
    tgt_x2 = px + tgt_r
    tgt_y2 = py + tgt_b

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

    intersection = (
        inter_w * inter_h
    )

    pred_area = (
        (pred_x2 - pred_x1).clamp(min=0)
        *
        (pred_y2 - pred_y1).clamp(min=0)
    )

    target_area = (
        (tgt_x2 - tgt_x1).clamp(min=0)
        *
        (tgt_y2 - tgt_y1).clamp(min=0)
    )

    union = (
        pred_area
        + target_area
        - intersection
    )

    iou = (
        intersection
        / union.clamp(min=1e-8)
    )

    return 1.0 - iou


def detection_loss(
    outputs,
    targets,
    image_size=224,
):
    """
    V3 anchor-free detection loss.

    Main changes from V2:

      1. Box distances are normalized by FPN stride.
      2. Box regression uses IoU loss.
      3. Classification is normalized by positive locations.
      4. Centerness uses a continuous target.
      5. Positive locations receive stronger learning signal.
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

            # Pixel coordinates of feature locations.
            px_pixels = (
                xx.float() + 0.5
            ) * stride

            py_pixels = (
                yy.float() + 0.5
            ) * stride

            # Normalized coordinates used by box_iou_loss.
            px = xx.float() + 0.5
            py = yy.float() + 0.5

            assigned_objects = 0

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

                # FPN scale assignment.
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

                # Normalize distances by stride.
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

                # FCOS-style centerness target.
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
                    ix
                ] = True

                assigned_objects += 1

            # --------------------------------------------------
            # Classification
            # --------------------------------------------------

            cls_loss_map = sigmoid_focal_loss(
                pred["cls"][b],
                cls_target,
            )

            num_positive = int(
                positive_mask.sum()
            )

            if num_positive > 0:

                positive_count = float(
                    num_positive
                )

                total_cls += (
                    cls_loss_map.sum()
                    / positive_count
                )

                # --------------------------------------------------
                # Box regression
                # --------------------------------------------------

                pos = (
                    positive_mask
                    .unsqueeze(0)
                    .expand_as(
                        pred["box"][b]
                    )
                )

                pred_boxes = (
                    pred["box"][b][pos]
                    .reshape(-1, 4)
                )

                target_boxes = (
                    box_target[pos]
                    .reshape(-1, 4)
                )

                pos_indices = (
                    positive_mask.nonzero(
                        as_tuple=False
                    )
                )

                pos_y = (
                    pos_indices[:, 0]
                )

                pos_x = (
                    pos_indices[:, 1]
                )

                point_x = (
                    pos_x.float() + 0.5
                )

                point_y = (
                    pos_y.float() + 0.5
                )

                total_box += (
                    box_iou_loss(
                        pred_boxes,
                        target_boxes,
                        point_x,
                        point_y,
                    ).mean()
                )

                # --------------------------------------------------
                # Centerness
                # --------------------------------------------------

                ctr_loss_map = (
                    F.binary_cross_entropy_with_logits(
                        pred["center"][b],
                        center_target,
                        reduction="none",
                    )
                )

                # Weight positive center locations.
                ctr_weights = torch.ones_like(
                    ctr_loss_map
                )

                ctr_weights[
                    center_target > 0
                ] = 4.0

                total_ctr += (
                    (
                        ctr_loss_map
                        * ctr_weights
                    ).sum()
                    / positive_count
                )

                total_positive += (
                    num_positive
                )

            else:

                # Keep background learning stable
                # when an FPN level has no assigned object.
                total_cls += (
                    cls_loss_map.mean()
                )

                total_ctr += (
                    F.binary_cross_entropy_with_logits(
                        pred["center"][b],
                        center_target,
                    )
                )

    normalizer = max(
        num_images,
        1,
    )

    total_cls = (
        total_cls / normalizer
    )

    total_box = (
        total_box / normalizer
    )

    total_ctr = (
        total_ctr / normalizer
    )

    # V3 loss weighting.
    loss = (
        total_cls
        +
        total_box
        +
        0.5 * total_ctr
    )

    return loss
