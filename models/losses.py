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

    if alpha >= 0:
        alpha_t = (
            alpha * targets
            + (1.0 - alpha) * (1.0 - targets)
        )
        focal_weight = (
            alpha_t * focal_weight
        )

    return (
        ce_loss * focal_weight
    )


def detection_loss(
    outputs,
    targets,
    image_size=224,
):
    """
    Anchor-free dense detection loss.

    Each object is assigned to one FPN level according
    to its area and to locations falling inside its box.

    Loss components:
      1. Focal classification loss
      2. L1 box regression loss
      3. Binary center/objectness loss
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

            px = (
                xx.float() + 0.5
            ) * stride

            py = (
                yy.float() + 0.5
            ) * stride

            positive_mask = torch.zeros(
                (h, w),
                dtype=torch.bool,
                device=device,
            )

            assigned_objects = 0

            for box, label in zip(
                boxes,
                labels,
            ):
                x1, y1, x2, y2 = box

                width = x2 - x1
                height = y2 - y1
                area = width * height

                # Scale ranges for the pyramid.
                if level == "p3":
                    valid_scale = area < (
                        64 * 64
                    )
                elif level == "p4":
                    valid_scale = (
                        area >= 64 * 64
                        and area < 128 * 128
                    )
                else:
                    valid_scale = area >= (
                        128 * 128
                    )

                if not valid_scale:
                    continue

                inside = (
                    (px >= x1)
                    & (px <= x2)
                    & (py >= y1)
                    & (py <= y2)
                )

                if not inside.any():
                    continue

                # Prefer the grid point closest to
                # the object's center.
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                distance = (
                    (px - cx) ** 2
                    + (py - cy) ** 2
                )

                distance = distance.masked_fill(
                    ~inside,
                    float("inf"),
                )

                flat_index = distance.argmin()

                iy = flat_index // w
                ix = flat_index % w

                cls_target[
                    int(label),
                    iy,
                    ix,
                ] = 1.0

                box_target[
                    0,
                    iy,
                    ix,
                ] = px[iy, ix] - x1

                box_target[
                    1,
                    iy,
                    ix,
                ] = py[iy, ix] - y1

                box_target[
                    2,
                    iy,
                    ix,
                ] = x2 - px[iy, ix]

                box_target[
                    3,
                    iy,
                    ix,
                ] = y2 - py[iy, ix]

                center_target[
                    0,
                    iy,
                    ix,
                ] = 1.0

                positive_mask[
                    iy,
                    ix,
                ] = True

                assigned_objects += 1

            # Classification.
            cls_element_loss = sigmoid_focal_loss(
                pred["cls"][b],
                cls_target,
            )

            total_cls += cls_element_loss.mean()

            # Center/objectness.
            ctr_loss = F.binary_cross_entropy_with_logits(
                pred["center"][b],
                center_target,
                reduction="none",
            )

            # Weight positive center locations more.
            ctr_weight = torch.ones_like(
                ctr_loss
            )

            ctr_weight[
                center_target == 1
            ] = 4.0

            total_ctr += (
                ctr_loss * ctr_weight
            ).mean()

            # Box regression only on positive points.
            if assigned_objects > 0:

                pos = positive_mask.unsqueeze(0)

                pred_boxes = pred["box"][b][
                    pos.expand_as(
                        pred["box"][b]
                    )
                ]

                target_boxes = box_target[
                    pos.expand_as(
                        box_target
                    )
                ]

                total_box += F.smooth_l1_loss(
                    pred_boxes,
                    target_boxes,
                    reduction="mean",
                )

    num_levels = 3
    normalizer = (
        num_images * num_levels
    )

    total_cls = (
        total_cls / normalizer
    )

    total_ctr = (
        total_ctr / normalizer
    )

    total_box = (
        total_box / normalizer
    )

    loss = (
        total_cls
        + total_box
        + 0.5 * total_ctr
    )

    return loss
