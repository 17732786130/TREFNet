import sys
import os
from pathlib import Path
import torch
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from PIL import Image
import cv2

# 添加项目根路径
current_path = os.path.abspath(__file__)
tools_dir = os.path.dirname(current_path)
project_root = os.path.dirname(tools_dir)
sys.path.insert(0, project_root)

# 导入项目模块
from models.model import BaseNet
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


# ======================== ✅ 路径校验（适配BCDD数据集结构） ========================
def check_dataset_path(dataset_root, split="test"):
    """校验数据集的test/train/val子目录结构"""
    A_dir = os.path.join(dataset_root, split, "A")
    B_dir = os.path.join(dataset_root, split, "B")
    label_dir = os.path.join(dataset_root, split, "label")

    print(f"\n🔍 校验 {split}集路径：")
    print(f"   数据集根目录：{dataset_root}")
    print(f"   A目录：{A_dir}")
    print(f"   B目录：{B_dir}")
    print(f"   Label目录：{label_dir}")

    paths_check = {"A目录": A_dir, "B目录": B_dir, "Label目录": label_dir}
    all_exists = True
    for name, path in paths_check.items():
        if os.path.exists(path):
            files = [name for name in os.listdir(path) if Path(name).suffix.lower() in IMAGE_EXTENSIONS]
            print(f"   ✅ {name}存在，有效文件数：{len(files)}")
            if len(files) == 0:
                print(f"   ⚠️ {name}无有效文件")
                all_exists = False
        else:
            print(f"   ❌ {name}不存在！")
            all_exists = False
    return all_exists, A_dir, B_dir, label_dir


# ======================== ✅ 适配BCDD数据集结构的Dataset类 ========================
class BCDDDataset(torch.utils.data.Dataset):
    def __init__(self, file_root, split="test", transform=None, manifest=None):
        self.file_root = file_root
        self.split = split
        self.transform = transform

        # 校验数据集路径
        is_valid, A_dir, B_dir, label_dir = check_dataset_path(file_root, split)
        if not is_valid:
            raise FileNotFoundError(f"{split}集路径无效：{file_root}")

        self.A_dir = A_dir
        self.B_dir = B_dir
        self.label_dir = label_dir

        # 获取有效文件（支持tif/png/jpg）
        self.sample_names = sorted(
            name for name in os.listdir(A_dir)
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(self.sample_names) == 0:
            raise ValueError(f"{split}集的A目录无有效文件：{A_dir}")

        if manifest is not None:
            names = manifest.read_text(encoding="utf-8-sig").splitlines()
            names = [name.strip() for name in names if name.strip()]
            if not names or len(set(names)) != len(names):
                raise ValueError(f"Empty list or duplicate entries: {manifest}")
            missing = set(names) - set(self.sample_names)
            if missing:
                raise ValueError(f"{manifest}: {len(missing)} missing images, e.g. {sorted(missing)[:5]}")
            self.sample_names = names

        label_names_by_stem = {}
        for label_name in os.listdir(self.label_dir):
            stem, extension = os.path.splitext(label_name)
            if extension.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
                label_names_by_stem.setdefault(stem, []).append(label_name)

        self.label_paths = {}
        for name in self.sample_names:
            if not os.path.isfile(os.path.join(self.B_dir, name)):
                raise FileNotFoundError(os.path.join(self.B_dir, name))
            label_path = os.path.join(self.label_dir, name)
            if not os.path.isfile(label_path):
                matches = label_names_by_stem.get(os.path.splitext(name)[0], [])
                if len(matches) != 1:
                    raise ValueError(f"Expected one mask for {name} in {self.label_dir}; found {matches}")
                label_path = os.path.join(self.label_dir, matches[0])
            self.label_paths[name] = label_path

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        name = self.sample_names[idx]

        # 读取影像和标签（BCDD标签：255=变化，0=未变化）
        img_A = cv2.imread(os.path.join(self.A_dir, name), cv2.IMREAD_COLOR)
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.imread(os.path.join(self.B_dir, name), cv2.IMREAD_COLOR)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.label_paths[name], cv2.IMREAD_GRAYSCALE)
        label = (label > 127).astype(np.float32)

        # 拼接A+B
        img = np.concatenate([img_A, img_B], axis=2)
        if self.transform is not None:
            img, label = self.transform(img, label)
        return img, label


# ======================== ✅ 精细化阈值搜索 ========================
def search_optimal_threshold(outputs, targets, search_range=(0.2, 0.8), step=0.01):
    best_f1 = -1.0
    best_threshold = 0.5
    count = round((search_range[1] - search_range[0]) / step) + 1
    thresholds = np.linspace(search_range[0], search_range[1], count)

    for threshold in thresholds:
        pred = torch.where(outputs > threshold,
                           torch.ones_like(outputs),
                           torch.zeros_like(outputs)).long()
        temp_meter = ConfuseMatrixMeter(n_class=2)
        f1 = temp_meter.update_cm(
            pr=pred.numpy().astype(np.int64),
            gt=targets.numpy().astype(np.int64)
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


# ======================== ✅ TTA增强变换（仅保留翻转） ========================
def tta_transforms(pre_img, post_img):
    tta_pairs = []
    # 1. 原始版本
    tta_pairs.append((pre_img, post_img))
    # 2. 水平翻转
    pre_flip_h = torch.flip(pre_img, dims=[3])
    post_flip_h = torch.flip(post_img, dims=[3])
    tta_pairs.append((pre_flip_h, post_flip_h))
    # 3. 垂直翻转
    pre_flip_v = torch.flip(pre_img, dims=[2])
    post_flip_v = torch.flip(post_img, dims=[2])
    tta_pairs.append((pre_flip_v, post_flip_v))
    return tta_pairs


# ======================== ✅ TTA逆变换 ========================
def tta_inverse(output, idx):
    if idx == 1:  # 水平翻转逆变换
        output = torch.flip(output, dims=[3])
    elif idx == 2:  # 垂直翻转逆变换
        output = torch.flip(output, dims=[2])
    return output


# ======================== ✅ 保存二值化可视化图 ========================
def save_binary_vis(pred_mask, sample_name, vis_dir, threshold):
    """
    保存二值化可视化预测图（黑/白）
    :param pred_mask: 模型预测mask (1, H, W)
    :param sample_name: 样本名称（和label一致）
    :param vis_dir: Vis文件夹路径
    :param threshold: 最优阈值
    """
    # 创建Vis目录
    os.makedirs(vis_dir, exist_ok=True)

    # 转换为numpy数组并二值化（>threshold=255(白), <=threshold=0(黑)）
    pred_np = pred_mask.squeeze().cpu().numpy()
    binary_mask = (pred_np > threshold).astype(np.uint8) * 255

    # 保存图片（格式和原样本一致）
    save_path = os.path.join(vis_dir, sample_name)
    Image.fromarray(binary_mask).save(save_path)

    return save_path


# ======================== ✅ 保存彩色可视化图（严格遵循颜色规则） ========================
def save_color_vis(pred_mask, target_mask, sample_name, color_vis_dir, threshold):
    """
    保存彩色可视化图（区分真实阳性/假阳性/真实阴性/假阴性）
    颜色规则：
    - 真实阳性（TP）：白色 (255,255,255)
    - 假阳性（FP）：红色 (255,0,0)
    - 真实阴性（TN）：黑色 (0,0,0)
    - 假阴性（FN）：蓝色 (0,0,255)
    :param pred_mask: 模型预测mask (1, H, W)
    :param target_mask: 真实标签mask (H, W)
    :param sample_name: 样本名称
    :param color_vis_dir: 彩色图保存目录
    :param threshold: 最优阈值
    """
    # 创建彩色图保存目录
    os.makedirs(color_vis_dir, exist_ok=True)

    # 预测图二值化（0/1）
    pred_np = pred_mask.squeeze().cpu().numpy()
    pred_binary = (pred_np > threshold).astype(np.uint8)
    # 真实标签归一化（0/1，兼容BCDD的255标签）
    target_np = target_mask.squeeze().cpu().numpy()
    target_binary = (target_np > 0.5).astype(np.uint8)

    # 初始化全黑背景（默认是真实阴性）
    h, w = pred_binary.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)  # 初始全黑=真实阴性

    # 1. 真实阳性（TP）：预测1，标签1 → 白色
    color_img[(pred_binary == 1) & (target_binary == 1)] = [255, 255, 255]
    # 2. 假阳性（FP）：预测1，标签0 → 红色
    color_img[(pred_binary == 1) & (target_binary == 0)] = [255, 0, 0]
    # 3. 假阴性（FN）：预测0，标签1 → 蓝色
    color_img[(pred_binary == 0) & (target_binary == 1)] = [0, 0, 255]
    # 4. 真实阴性（TN）：预测0，标签0 → 黑色（已默认初始化，无需额外赋值）

    # 保存彩色图
    save_path = os.path.join(color_vis_dir, sample_name)
    Image.fromarray(color_img).save(save_path)

    return save_path


# ======================== ✅ TTA推理函数：支持同时生成二值图+彩色图 ========================
@torch.no_grad()
def tta_ensemble_inference(args, model, test_loader, device, vis_dir=None, color_vis_dir=None, threshold=0.5):
    all_tta_outputs = []
    all_targets = []

    print(f"\n🚀 开始单模型TTA自融合推理...")
    print(f"📊 TTA增强版本数：3种（原始+水平翻转+垂直翻转）")
    print(f"📦 数据集样本数：{len(test_loader.dataset)}")

    # 获取样本名称列表（用于可视化命名）
    sample_names = test_loader.dataset.sample_names if (vis_dir or color_vis_dir) else None

    for batch_iter, batched_inputs in enumerate(test_loader):
        img, target = batched_inputs
        img = img.to(device).float()
        pre_img = img[:, 0:3].float()
        post_img = img[:, 3:6].float()
        target_var = target.to(device).float()

        # 生成TTA增强对
        tta_pairs = tta_transforms(pre_img, post_img)
        # 每个增强版本推理+逆变换
        tta_batch_outputs = []
        for idx, (pre_tta, post_tta) in enumerate(tta_pairs):
            probability = model(pre_tta, post_tta).probability
            tta_batch_outputs.append(tta_inverse(probability, idx))
        # 融合结果
        tta_ensemble_output = torch.stack(tta_batch_outputs, dim=0).mean(dim=0)

        # 保存可视化图（二值图+彩色图）
        if (vis_dir or color_vis_dir) and sample_names is not None:
            batch_size = tta_ensemble_output.shape[0]
            for idx_in_batch in range(batch_size):
                global_idx = batch_iter * args.batch_size + idx_in_batch
                if global_idx >= len(sample_names):
                    break
                # 1. 保存二值化图（如果指定了vis_dir）
                if vis_dir is not None:
                    save_binary_vis(
                        pred_mask=tta_ensemble_output[idx_in_batch],
                        sample_name=sample_names[global_idx],
                        vis_dir=vis_dir,
                        threshold=threshold
                    )
                # 2. 保存彩色图（如果指定了color_vis_dir）
                if color_vis_dir is not None:
                    save_color_vis(
                        pred_mask=tta_ensemble_output[idx_in_batch],
                        target_mask=target_var[idx_in_batch],
                        sample_name=sample_names[global_idx],
                        color_vis_dir=color_vis_dir,
                        threshold=threshold
                    )

        all_tta_outputs.append(tta_ensemble_output.detach().cpu())
        all_targets.append(target_var.cpu())

        if batch_iter % 10 == 0:
            print(f"\r[进度 {batch_iter + 1}/{len(test_loader)}]", end='', flush=True)

    all_tta_outputs = torch.cat(all_tta_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_tta_outputs, all_targets


def read_threshold(args, checkpoint):
    if args.threshold is not None:
        threshold = args.threshold
    else:
        path = args.threshold_file or checkpoint.with_name("best_threshold.txt")
        if not path.is_file():
            raise FileNotFoundError(
                f"Threshold file not found: {path}. "
                "Use --threshold or --threshold_file with a source-validation threshold."
            )
        threshold = float(path.read_text(encoding="utf-8-sig").strip())
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The threshold must be between 0 and 1.")
    return threshold


def main(args):
    if args.onGPU and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --onGPU false for CPU evaluation.")
    device = torch.device("cuda" if args.onGPU else "cpu")
    checkpoint = args.checkpoint.resolve()
    threshold = None if args.select_threshold else read_threshold(args, checkpoint)

    model = BaseNet(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {name[7:] if name.startswith("module.") else name: value for name, value in state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"The checkpoint does not match the 4.86 M TREF-Net model: {checkpoint}"
        ) from error
    model.to(device).eval()

    transform = myTransforms.Compose([
        myTransforms.Normalize(mean=[0.406, 0.456, 0.485] * 2,
                               std=[0.225, 0.224, 0.229] * 2),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor(),
    ])
    data = BCDDDataset(file_root=args.file_root, split=args.split, transform=transform, manifest=args.manifest)
    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir or checkpoint.parent / f"eval_{args.split}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    vis_dir = output / "Vis" if args.save_vis else None
    color_vis_dir = output / "ColorVis" if args.save_vis else None
    if vis_dir:
        vis_dir.mkdir()
    probabilities, targets = tta_ensemble_inference(
        args, model, loader, device, vis_dir, color_vis_dir, threshold,
    )
    if args.select_threshold:
        threshold, _ = search_optimal_threshold(probabilities, targets)
        (output / "best_threshold_tta.txt").write_text(f"{threshold:.6f}\n", encoding="utf-8")

    meter = ConfuseMatrixMeter(n_class=2)
    meter.update_cm(pr=(probabilities > threshold).long().numpy(), gt=targets.long().numpy())
    metrics = meter.get_scores()
    lines = [
        f"Checkpoint: {checkpoint}",
        f"Data: {args.file_root}",
        f"Split: {args.split}, samples: {len(data)}",
        "TTA: original, horizontal flip, vertical flip",
        f"Threshold: {threshold:.6f}",
    ]
    for display, key in (("F1", "F1"), ("IoU", "IoU"), ("Kappa", "Kappa"),
                         ("Recall", "recall"), ("Precision", "precision")):
        value = metrics.get(key)
        if value is not None:
            lines.append(f"{display}: {value:.6f}")
    result = "\n".join(lines)
    (output / "metrics.txt").write_text(result + "\n", encoding="utf-8")
    print(f"\n{result}\nOutput: {output}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate the 4.86 M TREF-Net model using three-view TTA.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--file_root", "--data-root", dest="file_root", required=True)
    parser.add_argument("--manifest", type=Path, help="Image filename list for the selected split")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    thresholds = parser.add_mutually_exclusive_group()
    thresholds.add_argument("--threshold", type=float)
    thresholds.add_argument("--threshold_file", "--threshold-file", type=Path)
    thresholds.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--inWidth", type=int, default=256)
    parser.add_argument("--inHeight", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--onGPU", default=True, type=lambda value: value.lower() == "true")
    args = parser.parse_args()
    if args.select_threshold and (args.split != "val" or args.save_vis):
        parser.error("--select-threshold requires --split val and cannot be combined with --save_vis")
    main(args)
