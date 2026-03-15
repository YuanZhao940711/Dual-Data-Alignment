import argparse
import os
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        parser.add_argument(
            "--name", type=str, default="experiment_name", help="Experiment name"
        )
        parser.add_argument(
            "--checkpoints_dir", type=str, required=True, help="Models are saved here"
        )
        parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="gpu ids: e.g. 0  0,1,2 use -1 for CPU",
        )
        parser.add_argument(
            "--num_threads", default=8, type=int, help="# threads for loading data"
        )
        parser.add_argument(
            "--batch_size", type=int, default=16, help="input batch size"
        )

        parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
        parser.add_argument(
            "--lora_alpha", type=float, default=1.0, help="LoRA scaling factor"
        )

        parser.add_argument(
            "--real_image_dir", default=None, help="Path to real images"
        )
        parser.add_argument(
            "--vae_image_dir", default=None, help="Path to VAE reconstructed images"
        )

        parser.add_argument(
            "--cropSize", type=int, default=336, help="Crop to this size"
        )

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            default = self.parser.get_default(k)
            comment = f"\t[default: {default}]" if v != default else ""
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # 修复：opt.txt 直接写入 checkpoints_dir，不再多拼接 name 子目录
        os.makedirs(opt.checkpoints_dir, exist_ok=True)
        with open(os.path.join(opt.checkpoints_dir, "opt.txt"), "wt") as opt_file:
            opt_file.write(message + "\n")

    def parse(self, print_options=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = [int(x) for x in str_ids if int(x) >= 0]

        # DDP 模式下，local_rank 由 torchrun 通过环境变量注入
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        opt.local_rank = local_rank
        opt.is_ddp = local_rank != -1

        if opt.is_ddp:
            # DDP 模式：每个进程只使用自己对应的 GPU
            opt.device = torch.device(f"cuda:{local_rank}")
        else:
            # 单卡模式
            if opt.gpu_ids:
                torch.cuda.set_device(opt.gpu_ids[0])
            opt.device = (
                torch.device(f"cuda:{opt.gpu_ids[0]}") if opt.gpu_ids
                else torch.device("cpu")
            )

        # 只在主进程打印和保存配置
        is_main = (not opt.is_ddp) or (local_rank == 0)
        if print_options and is_main:
            self.print_options(opt)

        self.opt = opt
        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        self.isTrain = True

        parser.add_argument("--niter", type=int, default=1, help="Total epochs")
        parser.add_argument(
            "--lr", type=float, default=1e-4, help="Initial learning rate"
        )
        parser.add_argument("--optim", type=str, default="adam", help="[sgd, adam]")
        parser.add_argument(
            "--accumulation_steps",
            type=int,
            default=1,
            help="Gradient accumulation steps",
        )
        parser.add_argument(
            "--weight_decay", type=float, default=0.0, help="L2 regularization"
        )

        parser.add_argument("--p_pixelmix", type=float, default=0.2)
        parser.add_argument("--r_pixelmix", type=float, default=0.0)

        parser.add_argument("--p_freqmix", type=float, default=0.2)
        parser.add_argument("--r_freqmix", type=float, default=0.1)

        parser.add_argument(
            "--quality_json",
            default="MSCOCO_train2017.json",
            help="JSON for quality settings",
        )

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--use_amp", action="store_true")
        parser.add_argument("--resume", type=int, default=0)
        parser.add_argument(
            "--grad_clip_norm", type=float, default=1.0, help="Max norm for gradients"
        )

        # ===== 损失权重 =====
        parser.add_argument(
            "--loss_cls_weight", type=float, default=0.5,
            help="Weight for BCE classification loss (0~1)"
        )
        parser.add_argument(
            "--loss_contrastive_weight", type=float, default=0.5,
            help="Weight for contrastive loss (0~1)"
        )

        # ===== 低误报率（Low FPR）控制 =====
        # pos_weight > 1：加重"将真实图判为AI生成"的惩罚，迫使模型偏向保守，
        # 使阈值天然偏高，从而降低误报率（FPR）。
        # 建议取值范围：1.0（无偏）~ 10.0（极低误报），车损/单证场景建议 3.0~5.0
        parser.add_argument(
            "--fp_penalty_weight", type=float, default=1.0,
            help="pos_weight for BCEWithLogitsLoss. >1 penalizes false positives "
                 "(real images predicted as fake) more heavily, reducing FPR. "
                 "Range: 1.0 (balanced) ~ 10.0 (near-zero FPR). "
                 "Recommended for insurance domain: 3.0~5.0"
        )

        # ===== LR Scheduler =====
        parser.add_argument(
            "--lr_T_max", type=int, default=0,
            help="T_max for CosineAnnealingLR. 0 = auto (set to total training steps)"
        )
        return parser
