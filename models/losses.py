import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits,
    targets,
    alpha=0.25,
    gamma=2.0,
):
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


def box_iou_and_l1_loss(
    pred_dist,
    target_dist,
    px,
    py,
):
    """
    Combined IoU + Smooth-L1 loss.

    Distances are normalized by the FPN stride.
    """

    pred_l = pred_dist[:, 0]
    pred_t = pred_dist[:, 1]
    pred_r = pred_dist[:, 2]
    pred_b = pred_dist[:, 3]

    tgt_l = target_dist[:, 0]
    tgt_t = target_dist[:, 1]
    tgt_r = target_dist[:, 2]
    tgt_b = target_dist[:, 3]

    # Predicted box.
    pred_x1 = px - pred_l
    pred_y1 = py - pred_t
    pred_x2 = px + pred_r
    pred_y2 = py + pred_b

    # Ground-truth box.
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

    iou_loss = 1.0 - iou

    l1_loss = F.smooth_l1_loss(
        pred_dist,
        target_dist,
        reduction="none",
    ).mean(dim=1)

    # IoU provides geometric overlap learning.
    # Smooth-L1 provides stable coordinate gradients
    # when overlap is initially poor.
    return (
        0.5 * iou_loss
        +
        0.5 * l1_loss
    )


def detection_loss(
    outputs,
    targets,
    image_size=224,
):
    """
    V4 anchor-free detection loss.

    Changes from V3:

      1. Multiple center-region locations are positive.
      2. Box loss combines IoU and Smooth-L1.
      3. Box distances remain stride-normalized.
      4. Centerness remains continuous.
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

            px_pixels = (
                xx.float() + 0.5
            ) * stride

            py_pixels = (
                yy.float() + 0.5
            ) * stride

            # --------------------------------------------------
            # Assign objects to FPN levels.
            # --------------------------------------------------

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

                # --------------------------------------------------
                # Center region.
                #
                # Instead of ONE positive location, use a central
                # region covering roughly 40% of the object.
                # --------------------------------------------------

                cx = (
                    x1 + x2
                ) / 2.0

                cy = (
                    y1 + y2
                ) / 2.0

                center_width = (
                    width * 0.4
                )

                center_height = (
                    height * 0.4
                )

                center_x1 = (
                    cx - center_width / 2.0
                )

                center_x2 = (
                    cx + center_width / 2.0
                )

                center_y1 = (
                    cy - center_height / 2.0
                )

                center_y2 = (
                    cy + center_height / 2.0
                )

                inside_center = (
                    (px_pixels >= center_x1)
                    &
                    (px_pixels <= center_x2)
                    &
                    (py_pixels >= center_y1)
                    &
                    (py_pixels <= center_y2)
                )

                # Fallback for very small objects.
                if not inside_center.any():

                    distances = (
                        (px_pixels - cx) ** 2
                        +
                        (py_pixels - cy) ** 2
                    )

                    flat = distances.argmin()

                    iy = flat // w
                    ix = flat % w

                    inside_center = torch.zeros(
                        (h, w),
                        dtype=torch.bool,
                        device=device,
                    )

                    inside_center[
                        iy,
                        ix
                    ] = True

                # --------------------------------------------------
                # Distances for ALL positive center locations.
                # --------------------------------------------------

                left = (
                    px_pixels - x1
                )

                top = (
                    py_pixels - y1
                )

                right = (
                    x2 - px_pixels
                )

                bottom = (
                    y2 - py_pixels
                )

                valid = (
                    inside_center
                    &
                    (left > 0)
                    &
                    (top > 0)
                    &
                    (right > 0)
                    &
                    (bottom > 0)
                )

                if not valid.any():
                    continue

                # Normalized box distances.
                box_target[
                    0,
                    valid
                ] = (
                    left[valid] / stride
                )

                box_target[
                    1,
                    valid
                ] = (
                    top[valid] / stride
                )

                box_target[
                    2,
                    valid
                ] = (
                    right[valid] / stride
                )

                box_target[
                    3,
                    valid
                ] = (
                    bottom[valid] / stride
                )

                cls_target[
                    int(label),
                    valid
                ] = 1.0

                # --------------------------------------------------
                # Continuous centerness.
                # --------------------------------------------------

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
                        /
                        lr_max.clamp(
                            min=1e-6
                        )
                    )
                    *
                    (
                        tb_min
                        /
                        tb_max.clamp(
                            min=1e-6
                        )
                    )
                )

                center_target[
                    0,
                    valid
                ] = centerness[valid]

                positive_mask |= valid

            # --------------------------------------------------
            # Classification loss.
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
                    /
                    positive_count
                )

                # --------------------------------------------------
                # Box regression.
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

                indices = (
                    positive_mask.nonzero(
                        as_tuple=False
                    )
                )

                pos_y = indices[:, 0]
                pos_x = indices[:, 1]

                point_x = (
                    pos_x.float() + 0.5
                )

                point_y = (
                    pos_y.float() + 0.5
                )

                total_box += (
                    box_iou_and_l1_loss(
                        pred_boxes,
                        target_boxes,
                        point_x,
                        point_y,
                    ).mean()
                )

                # --------------------------------------------------
                # Centerness loss.
                # --------------------------------------------------

                ctr_loss_map = (
                    F.binary_cross_entropy_with_logits(
                        pred["center"][b],
                        center_target,
                        reduction="none",
                    )
                )

                # Positive center locations receive more weight.
                ctr_weights = torch.ones_like(
                    ctr_loss_map
                )

                ctr_weights[
                    center_target > 0
                ] = 4.0

                total_ctr += (
                    (
                        ctr_loss_map
                        *
                        ctr_weights
                    ).sum()
                    /
                    positive_count
                )

            else:

                total_cls += (
                    cls_loss_map.mean()
                )

                total_ctr += (
                    F.binary_cross_entropy_with_logits(
                        pred["center"][b],
                        center_target,
                    )
                )

    total_cls /= max(
        num_images,
        1,
    )

    total_box /= max(
        num_images,
        1,
    )

    total_ctr /= max(
        num_images,
        1,
    )

    return (
        total_cls
        +
        total_box
        +
        0.5 * total_ctr
    )
