import os
import glob
import time
import yaml
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from data import create_dataloader
from networks.trainer import Trainer
from options import TrainOptions


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_latest_checkpoint(checkpoints_dir):
    latest = os.path.join(checkpoints_dir, "latest.pth")
    if os.path.exists(latest):
        return latest

    files = glob.glob(os.path.join(checkpoints_dir, "model_iters_*.pth"))
    if not files:
        return None

    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return files[-1]


def save_config(opt, checkpoints_dir):
    config_path = os.path.join(checkpoints_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(vars(opt), f, default_flow_style=False)


def is_main_process(opt):
    """判断是否为主进程（rank 0 或单卡模式）"""
    if getattr(opt, "is_ddp", False):
        return dist.get_rank() == 0
    return True


if __name__ == "__main__":
    opt = TrainOptions().parse()

    # ===== DDP 初始化 =====
    if getattr(opt, "is_ddp", False):
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(opt.local_rank)
        # DDP 模式下 per-GPU batch size 已由 create_dataloader 内部除以 world_size
        # 这里把 world_size 写入 opt 供后续使用
        opt.world_size = dist.get_world_size()
        opt.rank = dist.get_rank()
    else:
        opt.world_size = 1
        opt.rank = 0

    set_seed(getattr(opt, "seed", 42) + opt.rank)  # 不同进程用不同 seed，避免数据完全相同

    model = Trainer(opt)
    data_loader = create_dataloader(opt)

    # DDP 模式下替换 sampler 为 DistributedSampler
    if getattr(opt, "is_ddp", False):
        dataset = data_loader.dataset
        ddp_sampler = DistributedSampler(
            dataset,
            num_replicas=opt.world_size,
            rank=opt.rank,
            shuffle=True,
            drop_last=True,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=data_loader.batch_size,
            sampler=ddp_sampler,
            num_workers=data_loader.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=data_loader.collate_fn,
        )
    else:
        ddp_sampler = None

    # 只有主进程负责创建目录、写配置、开 TensorBoard
    if is_main_process(opt):
        os.makedirs(opt.checkpoints_dir, exist_ok=True)
        config_path = os.path.join(opt.checkpoints_dir, "config.yaml")
        if not os.path.exists(config_path):
            save_config(opt, opt.checkpoints_dir)
        writer = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir, "tensorboard"))
    else:
        writer = None

    start_time = time.time()
    if is_main_process(opt):
        print(f"Length of data loader: {len(data_loader)}")

    # 动态设置 CosineAnnealingLR 的 T_max = 总训练步数（仅当 lr_T_max=0 时生效）
    total_training_steps = len(data_loader) * opt.niter
    model.set_scheduler_T_max(total_training_steps)

    start_epoch = 0
    best_loss = float("inf")

    if int(opt.resume) == 1:
        latest_ckpt = find_latest_checkpoint(opt.checkpoints_dir)
        if latest_ckpt is not None:
            if is_main_process(opt):
                print(f"Resuming from checkpoint: {latest_ckpt}")
            resume_info = model.load_checkpoint(latest_ckpt)
            start_epoch = resume_info.get("epoch", 0)
            best_loss = resume_info.get("best_loss", float("inf"))
            if is_main_process(opt):
                print(
                    f"Resume success | start_epoch={start_epoch} | "
                    f"total_steps={model.total_steps} | best_loss={best_loss:.6f}"
                )
        else:
            if is_main_process(opt):
                print("No checkpoint found, starting from scratch.")

    for epoch in range(start_epoch, opt.niter):
        model.train()

        # DDP 模式下每个 epoch 需要 set_epoch 以保证 shuffle 不同
        if ddp_sampler is not None:
            ddp_sampler.set_epoch(epoch)

        epoch_loss_sum = 0.0
        epoch_steps = 0

        pbar = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=f"Epoch {epoch+1}/{opt.niter}",
            leave=True,
            dynamic_ncols=True,
            disable=not is_main_process(opt),  # 只有主进程显示进度条
        )

        for i, data in pbar:
            model.total_steps += 1
            model.set_input(data)
            step_metrics = model.optimize_parameters()

            loss_value = step_metrics["loss"]
            cls_loss_value = step_metrics["cls_loss"]
            contrastive_loss_value = step_metrics["contrastive_loss"]
            lr_value = model.optimizer.param_groups[0]["lr"]

            epoch_loss_sum += loss_value
            epoch_steps += 1
            avg_loss = epoch_loss_sum / epoch_steps

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                gpu_mem_str = f"{gpu_mem:.2f}G"
            else:
                gpu_mem_str = "CPU"

            if is_main_process(opt):
                pbar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    avg=f"{avg_loss:.4f}",
                    cls=f"{cls_loss_value:.4f}",
                    ctr=f"{contrastive_loss_value:.4f}",
                    lr=f"{lr_value:.2e}",
                    mem=gpu_mem_str,
                )

                if writer is not None:
                    writer.add_scalar("train/loss", loss_value, model.total_steps)
                    writer.add_scalar("train/loss_cls", cls_loss_value, model.total_steps)
                    writer.add_scalar("train/loss_contrastive", contrastive_loss_value, model.total_steps)
                    writer.add_scalar("train/lr", lr_value, model.total_steps)
                    if torch.cuda.is_available():
                        writer.add_scalar(
                            "train/gpu_mem_gb",
                            torch.cuda.memory_allocated() / 1024**3,
                            model.total_steps,
                        )

                # 周期性 checkpoint，只有主进程保存
                if model.total_steps % 5000 == 0:
                    elapsed = time.time() - start_time
                    avg_time_per_step = elapsed / max(model.total_steps, 1)
                    print(
                        f"\nStep: {model.total_steps} | "
                        f"Loss: {loss_value:.4f} | "
                        f"Avg Loss: {avg_loss:.4f} | "
                        f"Avg Time/Step: {avg_time_per_step:.4f}s"
                    )
                    model.save_checkpoint(
                        os.path.join(opt.checkpoints_dir, f"model_iters_{model.total_steps}.pth"),
                        epoch=epoch,
                        best_loss=best_loss,
                    )
                    model.save_checkpoint(
                        os.path.join(opt.checkpoints_dir, "latest.pth"),
                        epoch=epoch,
                        best_loss=best_loss,
                    )

        model.finalize_epoch()

        # 多卡下同步各进程的 epoch 平均 loss（取均值）
        if getattr(opt, "is_ddp", False):
            loss_tensor = torch.tensor(epoch_loss_sum / max(epoch_steps, 1)).to(opt.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            epoch_avg_loss = loss_tensor.item()
        else:
            epoch_avg_loss = epoch_loss_sum / max(epoch_steps, 1)

        if is_main_process(opt):
            if writer is not None:
                writer.add_scalar("epoch/avg_loss", epoch_avg_loss, epoch + 1)

            print(f"\nSaving model at end of epoch {epoch + 1}")
            model.save_checkpoint(
                os.path.join(opt.checkpoints_dir, f"model_epoch_{epoch + 1}.pth"),
                epoch=epoch + 1,
                best_loss=best_loss,
            )
            model.save_checkpoint(
                os.path.join(opt.checkpoints_dir, "latest.pth"),
                epoch=epoch + 1,
                best_loss=best_loss,
            )

            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                print(f"New best model saved with avg_loss={best_loss:.6f}")
                model.save_checkpoint(
                    os.path.join(opt.checkpoints_dir, "best.pth"),
                    epoch=epoch + 1,
                    best_loss=best_loss,
                )

    if writer is not None:
        writer.close()

    if is_main_process(opt):
        print("Training finished.")

    if getattr(opt, "is_ddp", False):
        dist.destroy_process_group()
