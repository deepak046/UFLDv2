import os
import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from utils.common import merge_config, get_model
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor as default_row_anchor, culane_col_anchor as default_col_anchor


def pred2coords(pred, row_anchor, col_anchor, local_width=1,
                original_image_width=2560, original_image_height=1440):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred["loc_row"].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred["loc_col"].shape

    max_indices_row = pred["loc_row"].argmax(1).cpu()
    valid_row = pred["exist_row"].argmax(1).cpu()

    max_indices_col = pred["loc_col"].argmax(1).cpu()
    valid_col = pred["exist_col"].argmax(1).cpu()

    pred["loc_row"] = pred["loc_row"].cpu()
    pred["loc_col"] = pred["loc_col"].cpu()

    coords = []

    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, max_indices_row[0, k, i] - local_width),
                                min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1,
                            )
                        )
                    )
                    out_tmp = (pred["loc_row"][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append((float(out_tmp), float(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(
                        list(
                            range(
                                max(0, max_indices_col[0, k, i] - local_width),
                                min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1,
                            )
                        )
                    )
                    out_tmp = (pred["loc_col"][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((float(col_anchor[k] * original_image_width), float(out_tmp)))
            coords.append(tmp)

    return coords


def parse_gt_lines_txt(path):
    lanes = []
    if not os.path.exists(path):
        return lanes
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = list(map(float, line.split()))
            pts = [(vals[2 * i], vals[2 * i + 1]) for i in range(len(vals) // 2)]
            lanes.append(pts)
    return lanes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/curvelanes_res18.py")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--index", type=int, default=0, help="Index of sample in valid set to inspect")
    args_cli = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    # Simulate command-line args for merge_config: it expects a positional 'config'
    import sys
    sys.argv = [sys.argv[0], args_cli.config]
    _, cfg = merge_config()

    assert cfg.dataset == "CurveLanes", "Use this debug script only for CurveLanes."

    img_transforms = transforms.Compose(
        [
            transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    list_file = os.path.join(cfg.data_root, "valid", "valid_for_culane_style.txt")
    dataset = LaneTestDataset(cfg.data_root, list_file, img_transform=img_transforms, crop_size=cfg.train_height)

    img, name = dataset[args_cli.index]
    print(f"Sample index: {args_cli.index}")
    print(f"Image name from list: {name}")

    net = get_model(cfg)
    state = torch.load(args_cli.checkpoint, map_location="cpu")
    state_dict = state.get("model", state)
    compatible_state = {}
    for k, v in state_dict.items():
        compatible_state[k[7:]] = v if k.startswith("module.") else v
    net.load_state_dict(compatible_state, strict=False)
    net.eval().cuda()

    with torch.no_grad():
        imgs = img.unsqueeze(0).cuda()
        pred = net(imgs)

    print("Pred tensor shapes:")
    for k in ["loc_row", "loc_col", "exist_row", "exist_col"]:
        v = pred.get(k, None)
        if v is not None:
            print(f"  {k}: {tuple(v.shape)}")

    row_anchor = getattr(cfg, "row_anchor", default_row_anchor)
    col_anchor = getattr(cfg, "col_anchor", default_col_anchor)
    pred_coords = pred2coords(
        pred, row_anchor, col_anchor, original_image_width=2560, original_image_height=1440
    )

    print("\nPredicted lanes (per-lane, first 5 points):")
    for li, lane in enumerate(pred_coords):
        print(f"  Lane {li}: {len(lane)} pts")
        print("    sample:", lane[:5])

    gt_rel = name.replace(".jpg", ".lines.txt")
    gt_path = os.path.join(cfg.data_root, gt_rel)
    gt_lanes = parse_gt_lines_txt(gt_path)
    print(f"\nGT file: {gt_path}")
    print(f"GT lanes: {len(gt_lanes)}")
    for li, lane in enumerate(gt_lanes):
        print(f"  GT lane {li}: {len(lane)} pts")
        print("    sample:", lane[:5])

    img_path = os.path.join(cfg.data_root, name)
    vis = cv2.imread(img_path)
    if vis is None:
        print(f"WARNING: could not read image {img_path}")
        return

    # GT and pred coords are in 2560x1440 space; rescale to actual image size
    img_h, img_w = vis.shape[:2]
    sx = img_w / 2560.0
    sy = img_h / 1440.0

    # Draw GT in red
    for lane in gt_lanes:
        for i in range(len(lane) - 1):
            p0 = (int(lane[i][0] * sx), int(lane[i][1] * sy))
            p1 = (int(lane[i + 1][0] * sx), int(lane[i + 1][1] * sy))
            cv2.line(vis, p0, p1, (0, 0, 255), 2)

    # Draw predictions in green
    for lane in pred_coords:
        for i in range(len(lane) - 1):
            p0 = (int(lane[i][0] * sx), int(lane[i][1] * sy))
            p1 = (int(lane[i + 1][0] * sx), int(lane[i + 1][1] * sy))
            cv2.line(vis, p0, p1, (0, 255, 0), 2)

    out_path = f"debug_overlay_{args_cli.index}.jpg"
    cv2.imwrite(out_path, vis)
    print(f"\nSaved overlay to {out_path} (GT=red, pred=green).")


if __name__ == "__main__":
    main()
