import sys
import os
import random

current_train_path = os.path.abspath(__file__)
tools_dir = os.path.dirname(current_train_path)
project_root = os.path.dirname(tools_dir)
sys.path.insert(0, project_root)

from models.model import TREF_Net, paper_loss

print(f"✅ 项目根目录已添加到搜索路径：{project_root}")

import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler

import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
import matplotlib.pyplot as plt

import time
import numpy as np
from argparse import ArgumentParser
from datetime import datetime


# ======================== ✅ 修复的损失函数（数值稳定） ========================
def _search_optimal_threshold(outputs, targets, mode="val", search_range=(0.3, 0.7), step=0.02):
    best_f1 = 0.0
    best_threshold = 0.5

    thresholds = np.arange(search_range[0], search_range[1] + step, step)

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


# ======================== 可视化工具函数 ========================
def tensor_to_numpy(tensor):
    if tensor.shape[1] == 3:
        mean = np.array([0.406, 0.456, 0.485])
        std = np.array([0.225, 0.224, 0.229])
    else:
        mean = np.array([0.406])
        std = np.array([0.225])

    np_arr = tensor.cpu().detach().numpy()
    np_arr = np.transpose(np_arr, (0, 2, 3, 1))
    np_arr = np_arr * std[None, None, None, :] + mean[None, None, None, :]
    np_arr = np.clip(np_arr, 0.0, 1.0)
    np_arr = np_arr.astype(np.float32)
    return np_arr


def make_image_grid(images):
    if len(images) == 0:
        raise ValueError("拼接失败：无有效图像（批次内无图像）")
    if images.ndim != 4:
        raise ValueError(f"输入图像格式错误：需 (B,H,W,C)，实际 {images.shape}")

    batch_size, h, w, c = images.shape
    grid = np.zeros((h * batch_size, w, c), dtype=np.float32)
    for i in range(batch_size):
        grid[i * h: (i + 1) * h, :, :] = images[i]
    return grid


def to_3channel(arr):
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    else:
        raise ValueError(f"不支持的图像维度：{arr.shape}（需2维单通道或3维1/3通道）")


# ======================== ✅ 早停机制 ========================
class EarlyStoppingTracker:
    def __init__(self, patience=50, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_f1 = float("-inf")
        self.best_epoch = 0

    def update(self, current_f1, epoch):
        if current_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = current_f1
            self.best_epoch = epoch
            self.counter = 0
            return True, False
        else:
            self.counter += 1
            should_stop = self.counter >= self.patience
            return False, should_stop


# ======================== ✅ 修复的训练函数 ========================
def train(args, train_loader, model, optimizer, epoch, max_batches, cur_iter=0, lr_factor=1.,
          vis_interval=200, log_interval=5):
    model.train()
    salEvalTrain = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total_batches = len(train_loader)
    max_iter_per_epoch = min(total_batches, args.max_iter_per_epoch)

    for batch_iter, batched_inputs in enumerate(train_loader):
        if batch_iter >= max_iter_per_epoch:
            break

        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        start_time = time.time()
        lr = adjust_learning_rate(args, optimizer, epoch, batch_iter + cur_iter, args.max_steps, lr_factor=lr_factor)

        if args.onGPU:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        probability = model(pre_img_var, post_img_var).probability

        loss = paper_loss(probability, target_var)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ Warning: 异常Loss={loss.item():.4f}，跳过本batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time

        pred = (probability.detach() > 0.5).long()

        with torch.no_grad():
            f1_batch = salEvalTrain.update_cm(
                pr=pred.cpu().numpy().astype(np.int64),
                gt=target_var.cpu().numpy().astype(np.int64)
            )

        if batch_iter % log_interval == 0:
            remaining_batches = args.max_steps - (batch_iter + cur_iter)
            res_time = remaining_batches * time_taken / 3600
            print(
                f'\r[迭代 {batch_iter + cur_iter:6d}/{args.max_steps:6d}] 批次 {batch_iter:3d}/{max_iter_per_epoch:3d} | '
                f'F1: {f1_batch:.3f} | LR: {lr:.2e} | Loss: {loss.data.item():.4f} | ETA: {res_time:.1f}h',
                end='', flush=True)

        if np.mod(batch_iter, vis_interval) == 1 and batch_iter > 0:
            try:
                batch_size = pre_img_var.shape[0]
                num_vis = min(2, batch_size)

                pre_img_np = tensor_to_numpy(pre_img_var[:num_vis])
                post_img_np = tensor_to_numpy(post_img_var[:num_vis])
                pred_np = tensor_to_numpy(pred[:num_vis].float())
                target_np = tensor_to_numpy(target_var[:num_vis])

                pre_grid = to_3channel(make_image_grid(pre_img_np))
                post_grid = to_3channel(make_image_grid(post_img_np))
                pred_grid = to_3channel(make_image_grid(pred_np))
                target_grid = to_3channel(make_image_grid(target_np))

                vis = np.concatenate([pre_grid, post_grid, pred_grid, target_grid], axis=1)
                file_name = os.path.join(args.vis_dir, f'train_ep{epoch:03d}_iter{batch_iter:03d}.jpg')
                plt.imsave(file_name, vis)

            except Exception as e:
                pass

        if batch_iter + cur_iter >= args.max_steps:
            break

    scores_train = salEvalTrain.get_scores()
    train_f1 = scores_train.get('F1', 0.0)
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0

    print(f"\n  ✓ Train F1={train_f1:.4f} | Loss={average_epoch_loss_train:.4f}")

    return average_epoch_loss_train, scores_train, lr, train_f1, 0.5


# ======================== ✅ 验证函数 ========================
@torch.no_grad()
def val(args, val_loader, model, epoch, mode="val", fixed_threshold=None):
    if mode == "test" and fixed_threshold is None:
        raise ValueError("Test evaluation requires the threshold selected on validation data.")
    model.eval()
    epoch_loss = []
    total_batches = len(val_loader)

    all_outputs = []
    all_targets = []

    for batch_iter, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        if args.onGPU:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        probability = model(pre_img_var, post_img_var).probability

        loss = paper_loss(probability, target_var)

        epoch_loss.append(loss.data.item())
        all_outputs.append(probability.detach().cpu())
        all_targets.append(target_var.cpu())

        if batch_iter % 5 == 0:
            print(f'\r  [{batch_iter:3d}/{total_batches:3d}] Loss: {loss.data.item():.4f}',
                  end='', flush=True)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    if fixed_threshold is None:
        best_threshold, _ = _search_optimal_threshold(
            all_outputs, all_targets, mode=mode, search_range=(0.3, 0.7), step=0.02
        )
    else:
        best_threshold = fixed_threshold

    pred_best = torch.where(all_outputs > best_threshold,
                            torch.ones_like(all_outputs),
                            torch.zeros_like(all_outputs)).long()

    final_meter = ConfuseMatrixMeter(n_class=2)
    final_meter.update_cm(
        pr=pred_best.numpy().astype(np.int64),
        gt=all_targets.numpy().astype(np.int64)
    )
    best_metrics = final_meter.get_scores()
    final_f1 = best_metrics.get('F1', 0.0)

    average_epoch_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0

    print(
        f"\n  ✓ {mode.upper()} F1={final_f1:.4f} (threshold={best_threshold:.3f}) | Loss={average_epoch_loss:.4f}")

    return average_epoch_loss, best_metrics, final_f1, best_threshold


# ======================== 学习率调整函数 ========================
def adjust_learning_rate(args, optimizer, epoch, iter, max_iter, lr_factor=1):
    warmup_steps = 200
    if iter < warmup_steps:
        lr = args.lr * (iter + 1) / warmup_steps
    else:
        cur_iter = iter - warmup_steps
        max_iter_adjusted = max_iter - warmup_steps
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter_adjusted) ** 0.9

    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# ======================== 适配BCDD数据集的自定义Dataset类 ========================
class BCDDDataset(torch.utils.data.Dataset):
    """适配BCDD数据集结构的Dataset类（train/val/test下直接包含A/B/label）"""

    def __init__(self, split, file_root, transform=None):
        self.split = split
        self.file_root = file_root
        self.transform = transform

        # 构建A/B/label的路径（BCDD的结构是 split/A, split/B, split/label）
        self.A_dir = os.path.join(file_root, split, "A")
        self.B_dir = os.path.join(file_root, split, "B")
        self.label_dir = os.path.join(file_root, split, "label")

        # 获取所有文件名（确保A/B/label的文件名完全一致）
        self.sample_names = sorted(
            name for name in os.listdir(self.A_dir)
            if os.path.splitext(name)[1].lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        )
        if not self.sample_names:
            raise ValueError(f"No paired images found in {self.A_dir}")

        # WHU provides TIFF images and PNG masks with the same stem.
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
        import cv2

        name = self.sample_names[idx]

        # 读取A期影像
        img_A = cv2.imread(os.path.join(self.A_dir, name), cv2.IMREAD_COLOR)
        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)

        # 读取B期影像
        img_B = cv2.imread(os.path.join(self.B_dir, name), cv2.IMREAD_COLOR)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        # 读取标签
        label = cv2.imread(self.label_paths[name], cv2.IMREAD_GRAYSCALE)
        label = (label > 127).astype(np.float32)

        # 拼接A和B
        img = np.concatenate([img_A, img_B], axis=2)

        # 应用变换
        if self.transform is not None:
            img, label = self.transform(img, label)

        return img, label


# ======================== 主训练函数 ========================
def trainValidateSegmentation(args):
    if args.onGPU and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --onGPU false for CPU training.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.onGPU:
        torch.cuda.manual_seed_all(args.seed)

    model = TREF_Net(pretrained=args.pretrained)

    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    args.logFile = args.logFile or f"trainValLog_{dataset_name}.txt"
    args.file_root = os.path.abspath(args.file_root)
    args.savedir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(
        project_root, "runs_trefnet_486",
        f"{dataset_name}_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}",
    )
    if os.path.exists(args.savedir) and os.listdir(args.savedir):
        raise FileExistsError(f"Choose an empty output directory: {args.savedir}")
    args.vis_dir = os.path.join(args.savedir, "Vis")

    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"📁 结果保存到：{args.savedir}")
    print(f"🖼️  可视化保存到：{args.vis_dir}")
    print(f"{'=' * 80}\n")

    if args.onGPU:
        model = model.cuda()
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"🧠 模型参数：{total_params:,}\n")

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    print(f"📥 加载数据集...")
    # ======================== ✅ 使用适配BCDD结构的Dataset ========================
    train_data = BCDDDataset(
        split="train",
        file_root=args.file_root,
        transform=trainDataset_main
    )
    trainLoader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True
    )

    val_data = BCDDDataset(
        split="val",
        file_root=args.file_root,
        transform=valDataset
    )
    valLoader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False
    )

    test_data = BCDDDataset(
        split="test",
        file_root=args.file_root,
        transform=valDataset
    )
    testLoader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False
    )

    max_batches = len(trainLoader)
    if max_batches == 0:
        raise ValueError("No complete training batches. Reduce --batch_size.")
    args.max_iter_per_epoch = min(args.max_iter_per_epoch, max_batches)
    args.max_steps = args.max_epochs * args.max_iter_per_epoch
    print(f"✅ 数据加载完成：Train {len(train_data)} | Val {len(val_data)} | Test {len(test_data)}")
    print(f"📅 计划：{args.max_epochs} epochs × {args.max_iter_per_epoch} iters/epoch\n")

    args.resume = False
    args.resume_path = os.path.join(args.savedir, "checkpoint.pth.tar")

    logFileLoc = os.path.join(args.savedir, args.logFile)
    if not os.path.isfile(logFileLoc):
        with open(logFileLoc, 'w', encoding='utf-8') as f:
            f.write(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"参数数量：{total_params:,}\n")
            f.write(f"最大Epochs：{args.max_epochs}\n")
            f.write(f"早停耐心值：{args.early_stopping_patience}\n")
            f.write(f"批次大小：{args.batch_size}\n")
            f.write(f"学习率：{args.lr}\n")
            f.write("=" * 80 + "\n")
            f.write("Epoch\tTrain_F1\tVal_F1\tVal_Threshold\tTrain_Loss\tVal_Loss\tLR\n")

    global_max_F1_val = 0.0
    global_max_F1_train = 0.0
    best_val_threshold_saved = 0.5

    early_stopping = EarlyStoppingTracker(patience=args.early_stopping_patience, min_delta=1e-6)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-7
    )

    print(f"🔥 开始训练 ({args.max_epochs} epochs)...")
    print("=" * 80)

    start_time = time.time()
    cur_iter = 0

    try:
        for epoch in range(args.max_epochs):
            epoch_start_time = time.time()
            print(f"\n[Epoch {epoch + 1:3d}/{args.max_epochs}]")
            print("-" * 80)

            lossTr, score_tr, current_lr, train_f1, _ = train(
                args, trainLoader, model, optimizer, epoch, max_batches, cur_iter
            )
            cur_iter += args.max_iter_per_epoch

            torch.cuda.empty_cache()

            lossVal, score_val, val_f1, best_val_threshold = val(
                args, valLoader, model, epoch, mode="val"
            )
            torch.cuda.empty_cache()

            scheduler.step(val_f1)

            is_best, should_stop = early_stopping.update(val_f1, epoch)

            if is_best:
                global_max_F1_val = val_f1
                global_max_F1_train = train_f1
                best_val_threshold_saved = best_val_threshold
                best_model_path = os.path.join(args.savedir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                with open(os.path.join(args.savedir, "best_threshold.txt"), "w", encoding="utf-8") as threshold_file:
                    threshold_file.write(f"{best_val_threshold_saved:.6f}\n")
                print(f"  🏆 最优F1更新: Val_F1={val_f1:.4f}, Train_F1={train_f1:.4f}")

            with open(logFileLoc, 'a', encoding='utf-8') as logger:
                logger.write(
                    f"{epoch + 1}\t{train_f1:.4f}\t{val_f1:.4f}\t{best_val_threshold:.4f}\t"
                    f"{lossTr:.4f}\t{lossVal:.4f}\t{current_lr:.2e}\n"
                )

            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_f1': val_f1,
                'train_f1': train_f1,
            }, args.resume_path)

            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            estimated_total = total_time * args.max_epochs / (epoch + 1)
            remaining_time = estimated_total - total_time

            print(f"  耐心值: {early_stopping.counter}/{args.early_stopping_patience} | "
                  f"本Epoch耗时: {epoch_time / 60:.1f}min | 剩余估计: {remaining_time / 3600:.1f}h")

            if should_stop:
                print(f"\n⚠️  早停触发（{args.early_stopping_patience} epochs无改进）")
                print(f"最好的Epoch是第 {early_stopping.best_epoch + 1} 个，F1={early_stopping.best_f1:.4f}")
                break

    except KeyboardInterrupt:
        print(f"\n⚠️  训练被中断")
        return

    print(f"\n{'=' * 80}")
    print(f"🎉 训练完成！评估测试集...")
    print(f"{'=' * 80}")

    best_model_path = os.path.join(args.savedir, "best_model.pth")
    if os.path.isfile(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location="cpu", weights_only=True), strict=True)
        print(f"✅ 加载最优模型")

    loss_test, score_test, test_f1, test_threshold = val(
        args, testLoader, model, epoch=args.max_epochs, mode="test",
        fixed_threshold=best_val_threshold_saved,
    )

    print(f"\n{'=' * 80}")
    print(f"📊 测试集结果")
    print(f"{'=' * 80}")
    print(f"Train F1: {global_max_F1_train:.4f}")
    print(f"Val F1:   {global_max_F1_val:.4f} (threshold={best_val_threshold_saved:.3f})")
    print(f"Test F1:  {test_f1:.4f} (threshold={test_threshold:.3f})")

    test_kappa = score_test.get('Kappa', 0.0)
    test_iou = score_test.get('IoU', 0.0)
    test_recall = score_test.get('recall', 0.0)
    test_precision = score_test.get('precision', 0.0)

    print(f"\n📋 测试集详细指标:")
    print(f"  - Kappa: {test_kappa:.4f}")
    print(f"  - IoU: {test_iou:.4f}")
    print(f"  - Recall: {test_recall:.4f}")
    print(f"  - Precision: {test_precision:.4f}")

    total_training_time = time.time() - start_time
    print(f"\n⏱️  总训练耗时: {total_training_time / 3600:.1f}h")
    print(f"📁 日志: {logFileLoc}")
    print(f"🤖 模型: {best_model_path}")
    print("=" * 80)


# ======================== 入口函数 ========================
if __name__ == '__main__':
    parser = ArgumentParser(description="TREF-Net 4.86 M 变化检测")
    parser.add_argument("--file_root", "--data-root", required=True, help="Root containing train, val, and test folders")
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--output_dir', default=None, help='Output directory for the new model')
    parser.add_argument('--pretrained', action='store_true', help='Initialize MobileNetV2 with ImageNet weights')
    parser.add_argument('--inWidth', type=int, default=256)
    parser.add_argument('--inHeight', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--step_loss', type=int, default=15)
    parser.add_argument('--lr', type=float, default=5e-4)   
    parser.add_argument('--lr_mode', default='poly', choices=['poly', 'step'])
    parser.add_argument('--logFile', default=None, help='Training log filename')
    parser.add_argument('--onGPU', default=True, type=lambda x: str(x).lower() == 'true')
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--max_iter_per_epoch', type=int, default=100)
    parser.add_argument('--early_stopping_patience', type=int, default=300)

    args = parser.parse_args()
    args.max_steps = args.max_epochs * args.max_iter_per_epoch

    print("\n" + "=" * 80)
    print("🚀 TREF-Net 4.86 M 变化检测训练")
    print("=" * 80)
    print(f"{'参数名':<25} | {'参数值':<30}")
    print("-" * 80)
    for arg in vars(args):
        value = getattr(args, arg)
        print(f"{arg:<25} | {str(value):<30}")
    print("=" * 80 + "\n")

    trainValidateSegmentation(args)