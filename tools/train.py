import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
import wandb

# RUN
#  bash scripts/dist_train.sh 4 --cfg_file ./cfgs/dsvt_models/dsvt_plain_1f_onestage_SL.yaml --sync_bn --logger_iter_interval 500 --batch_size 40 --epochs 1000 --start_epoch 5
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/home/shchoi/workspace/DSVT/tools/cfgs/dsvt_models/dsvt_plain_1f_onestage.yaml', help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=4, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=200, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=20, help='number of training epochs')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=True, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    

    parser.add_argument('--fp16', action='store_true', default=False, help='trigger mixed precision')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    # 분산/단일 설정
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils,
            f'init_dist_{args.launcher}')(args.tcp_port, args.local_rank, backend='nccl')
        dist_train = True

    # 배치 및 에폭 설정
    args.batch_size = args.batch_size or cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    assert args.batch_size % total_gpus == 0, 'Batch size must be divisible by GPUs'
    args.batch_size //= total_gpus
    args.epochs = args.epochs or cfg.OPTIMIZATION.NUM_EPOCHS

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    # 출력 디렉토리
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 로거
    log_file = output_dir / f'log_train_{datetime.datetime.now():%Y%m%d-%H%M%S}.txt'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    logger.info('**********************Start logging**********************')
    logger.info(f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES","ALL")}')
    for k,v in vars(args).items(): logger.info(f'{k:16} {v}')
# ─── TensorBoard 및 W&B 초기화 ─────────────────────────────────────────────
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
    if cfg.LOCAL_RANK == 0:
        wandb.init(
            project='DSVT',
            name=f'{cfg.EXP_GROUP_PATH}_{cfg.TAG}_{args.extra_tag}',
            config=cfg,
            sync_tensorboard=True,  # 이 옵션만으로 자동 패치됩니다
        )

        # add_scalar 오버라이드 (TensorBoard & W&B 동시 로깅)
        original_add_scalar = tb_log.add_scalar
        def add_scalar_and_wandb(tag, value, step=None, walltime=None):
            original_add_scalar(tag, value, step, walltime)
            wandb.log({tag: value}, step=step)
        tb_log.add_scalar = add_scalar_and_wandb

# ※ 여기서 wandb.tensorboard.patch() 호출은 제거합니다.
# ─────────────────────────────────────────────────────────────────────────────    # 데이터로더
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    # 모델 생성 및 GPU 이동
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    if cfg.LOCAL_RANK == 0:
        wandb.watch(model, log='all')

    # DDP 래핑
    if dist_train:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()]
        )
    logger.info(model)

    # 옵티마이저, 스케줄러
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    lr_scheduler, lr_warmup = build_scheduler(
        optimizer,
        total_iters_each_epoch=len(train_loader),
        total_epochs=args.epochs,
        last_epoch=0,
        optim_cfg=cfg.OPTIMIZATION
    )

    # 트레이닝
    logger.info(f'**********************Start training {cfg.TAG}**********************')
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=0,
        total_epochs=args.epochs,
        start_iter=0,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record,
        show_gpu_stat=not args.wo_gpu_stat,
        fp16=args.fp16,
        cfg=cfg
    )

    # 평가
    logger.info(f'**********************Start evaluation {cfg.TAG}**********************')
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=False
    )
    eval_dir = output_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader,
        args,
        eval_dir,
        logger,
        ckpt_dir,
        dist_test=dist_train
    )
    logger.info(f'**********************End evaluation {cfg.TAG}**********************')

if __name__ == '__main__':
    main()
