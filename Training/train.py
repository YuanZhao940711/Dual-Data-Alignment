import os
import glob
import time
import yaml
import random
import numpy as np
import torch
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


if __name__ == "__main__":
    opt = TrainOptions().parse()

    set_seed(getattr(opt, "seed", 42))

    model = Trainer(opt)
    data_loader = create_dataloader(opt)

    os.makedirs(opt.checkpoints_dir, exist_ok=True)
    # save_config(opt, opt.checkpoints_dir)
    config_path = os.path.join(opt.checkpoints_dir, "config.yaml")
    if not os.path.exists(config_path):
        save_config(opt, opt.checkpoints_dir)
        
    writer = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir, "tensorboard"))

    start_time = time.time()
    print(f"Length of data loader: {len(data_loader)}")

    start_epoch = 0
    best_loss = float("inf")

    if int(opt.resume) == 1:
        latest_ckpt = find_latest_checkpoint(opt.checkpoints_dir)
        if latest_ckpt is not None:
            print(f"Resuming from checkpoint: {latest_ckpt}")
            resume_info = model.load_checkpoint(latest_ckpt)

            start_epoch = resume_info.get("epoch", 0)
            best_loss = resume_info.get("best_loss", float("inf"))
            print(
                f"Resume success | start_epoch={start_epoch} | total_steps={model.total_steps} | best_loss={best_loss:.6f}"
            )
        else:
            print("No checkpoint found, starting from scratch.")


    for epoch in range(start_epoch, opt.niter):
        model.train()

        epoch_loss_sum = 0.0
        epoch_steps = 0

        # tqdm progress bar
        pbar = tqdm(
            enumerate(data_loader), 
            total=len(data_loader), 
            desc=f"Epoch {epoch+1}/{opt.niter}",
            leave=True,
            dynamic_ncols=True,
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

            # update tqdm info
            pbar.set_postfix(
                loss=f"{loss_value:.4f}",
                avg=f"{avg_loss:.4f}",
                cls=f"{cls_loss_value:.4f}",
                ctr=f"{contrastive_loss_value:.4f}",
                lr=f"{lr_value:.2e}",
                mem=gpu_mem_str,
            )
            
            # tensorboard logging
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

            # periodic checkpoint
            if model.total_steps % 5000 == 0:
                elapsed = time.time() - start_time
                avg_time_per_step = elapsed / max(model.total_steps, 1)

                print(
                    f"\nStep: {model.total_steps} | "
                    f"Loss: {loss_value:.4f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Avg Time/Step: {avg_time_per_step:.4f}s"
                )
                # model.save_networks(f"model_iters_{model.total_steps}.pth")

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

        epoch_avg_loss = epoch_loss_sum / max(epoch_steps, 1)
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

        # best model saving
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            print(f"New best model saved with avg_loss={best_loss:.6f}")
            model.save_checkpoint(
                os.path.join(opt.checkpoints_dir, "best.pth"),
                epoch=epoch + 1,
                best_loss=best_loss,
            )
        
        # print(f"Saving model at end of epoch {epoch}")
        # model.save_networks(f"model_epoch_{epoch}.pth")
    
    writer.close()
    print("Training finished.")
