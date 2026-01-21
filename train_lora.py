import argparse, os, datetime, logging, time, math, numpy as np, warnings, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn

from core.configs import cfg
from core.datasets import build_dataset
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger

# <<< 关键：导入你刚添加的 DINOv2 + Adapter 分割模型 >>>
from core.models.dinov2_seg_adapter import DINOv2SegPPM

warnings.filterwarnings('ignore')

# -------- AMP 兼容 --------
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def amp_backward(loss, optimizer, retain_graph=False):
    if APEX_AVAILABLE:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(retain_graph=retain_graph)
    else:
        loss.backward(retain_graph=retain_graph)

# -------- strip_prefix（保留以便 resume 兼容）--------
def strip_prefix_if_present(state_dict, prefix):
    from collections import OrderedDict
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix + 'layer5'):
            continue
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

# ===================== 训练 =====================
def train(cfg, local_rank, distributed):
    logger = logging.getLogger("FADA.trainer")
    logger.info("Start training (DINOv2 + Adapter + SegHead)")

    device = torch.device(cfg.MODEL.DEVICE)

    # ---- 读取可选超参（若 cfg 没有对应字段则使用默认值）----
    get = lambda group, key, default: getattr(getattr(cfg, group, type("obj",(object,),{})()), key, default)
    model_name      = get("DINOV2ADAPTER", "MODEL_NAME", "dinov2_vitl14")
    hub_dir         = get("DINOV2ADAPTER", "HUB_DIR", "dinov2-main/")
    num_classes     = cfg.MODEL.NUM_CLASSES
    lora_r          = get("DINOV2ADAPTER", "LORA_R", 8)
    lora_alpha      = get("DINOV2ADAPTER", "LORA_ALPHA", 16)
    lora_dropout    = get("DINOV2ADAPTER", "LORA_DROPOUT", 0.0)
    freeze_backbone = get("DINOV2ADAPTER", "FREEZE_BACKBONE", True)
    # 学习率倍率（分组）
    head_lr_mult    = get("DINOV2ADAPTER", "HEAD_LR_MULT", 1.0)
    lora_lr_mult    = get("DINOV2ADAPTER", "LORA_LR_MULT", 0.4)

    # ---- 构建单模型：DINOv2 + LoRA + PPMHead ----
    model = DINOv2SegPPM(
        hub_dir=hub_dir,
        model_name=model_name,
        num_classes=num_classes,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        freeze_backbone=freeze_backbone
    ).to(device)

    # ---- DDP 包装 ----
    batch_size = cfg.SOLVER.BATCH_SIZE
    if distributed:
        pg = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size())
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()

    # ---- 优化器（分组：seg head 与 LoRA）----
    head_params, lora_params = [], []
    named_params = model.named_parameters() if not distributed else model.module.named_parameters()
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if (".A." in n) or (".B." in n):   # LoRA 参数
            lora_params.append(p)
        else:
            head_params.append(p)

    base_lr = cfg.SOLVER.BASE_LR
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": base_lr * head_lr_mult, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
        {"params": lora_params, "lr": base_lr * lora_lr_mult, "weight_decay": 0.0},
    ])

    # ---- AMP（可选）----
    if APEX_AVAILABLE:
        if distributed:
            [model.module], [optimizer] = amp.initialize([model.module], [optimizer],
                                                         opt_level="O2", keep_batchnorm_fp32=True, loss_scale="dynamic")
        else:
            [model], [optimizer] = amp.initialize([model], [optimizer],
                                                 opt_level="O2", keep_batchnorm_fp32=True, loss_scale="dynamic")

    # ---- Resume（可选）----
    if getattr(cfg, "resume", ""):
        logger.info(f"Loading checkpoint from {cfg.resume}")
        checkpoint = torch.load(cfg.resume, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        state = strip_prefix_if_present(state, "module.")
        (model.module if distributed else model).load_state_dict(state, strict=False)

    # ---- 数据 ----
    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data) if distributed else None
    train_loader = torch.utils.data.DataLoader(
        src_train_data,
        batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True
    )

    # ---- 损失 ----
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # ---- 日志&工具 ----
    def debug_param_stats(model_):
        tot = train = lora = 0
        for n, p in model_.named_parameters():
            num = p.numel()
            tot += num
            if p.requires_grad:
                train += num
                if (".A." in n) or (".B." in n):
                    lora += num
        print(f"[ParamStats] total={tot/1e6:.2f}M, trainable={train/1e6:.2f}M, lora={lora/1e6:.2f}M")

    def _is_main_process():
        return (not distributed) or (
            torch.distributed.is_available() and torch.distributed.is_initialized()
            and torch.distributed.get_rank() == 0
        )

    def _extract_val_score(val_result):
        """从 run_test 的返回中提取验证分数（mIoU/score等）。兼容 float 或 dict。"""
        if val_result is None:
            return None
        if isinstance(val_result, (int, float)):
            return float(val_result)
        if isinstance(val_result, dict):
            for k in ['mIoU', 'miou', 'mean_iou', 'meanIoU', 'val_mIoU', 'score', 'main', 'Main']:
                if k in val_result:
                    try:
                        return float(val_result[k])
                    except Exception:
                        pass
            summ = val_result.get('summary') if isinstance(val_result.get('summary'), dict) else None
            if summ is not None:
                for k in ['mIoU', 'miou', 'mean_iou', 'meanIoU', 'score']:
                    if k in summ:
                        try:
                            return float(summ[k])
                        except Exception:
                            pass
        return None

    # ---- 训练主循环 ----
    max_iters = cfg.SOLVER.MAX_ITER
    meters = MetricLogger(delimiter="  ")
    (model.module if distributed else model).train()
    start_training_time = time.time()
    end = time.time()
    iteration = 0
    output_dir = cfg.OUTPUT_DIR

    # 准备输出目录 & best/last 路径
    if _is_main_process() and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    best_score = float('-inf')
    best_iter  = -1
    best_path  = os.path.join(output_dir, 'model_best.pth') if output_dir else None
    last_path  = os.path.join(output_dir, 'model_last.pth') if output_dir else None

    # 打印参数规模
    m = model.module if distributed else model
    debug_param_stats(m)

    for i, (src_input, src_label, _) in enumerate(train_loader):
        data_time = time.time() - end

        # 学习率调度
        current_lr = adjust_learning_rate(
            cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters, power=cfg.SOLVER.LR_POWER
        )
        optimizer.param_groups[0]['lr'] = current_lr * head_lr_mult
        optimizer.param_groups[1]['lr'] = current_lr * lora_lr_mult

        optimizer.zero_grad(set_to_none=True)
        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True).long()

        # 前向
        out = (model.module if distributed else model)(src_input)  # {'logits': [B,C,H,W]}
        logits = out['logits']
        loss_seg = criterion(logits, src_label)

        # 反传 & 更新
        amp_backward(loss_seg, optimizer)
        optimizer.step()

        # 统计
        meters.update(loss=loss_seg.item())
        iteration += 1

        # 日志
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iters - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % 20 == 0 or iteration == max_iters:
            logging.getLogger("FADA.trainer").info(
                meters.delimiter.join(
                    ["eta: {eta}", "iter: {iter}", "{meters}",
                     "lr(head): {lrh:.6f}", "lr(lora): {lrl:.6f}",
                     "max mem: {memory:.0f}", "best: {best:.4f}@{best_it}"]
                ).format(
                    eta=eta_string, iter=iteration, meters=str(meters),
                    lrh=optimizer.param_groups[0]["lr"], lrl=optimizer.param_groups[1]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    best=(best_score if best_score != float('-inf') else float('nan')),
                    best_it=best_iter
                )
            )

        # ===== 验证（只在取得最好成绩时保存 best，不保存中间权重） =====
        need_eval = (iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0) or (iteration == cfg.SOLVER.STOP_ITER)
        if need_eval and (not distributed or _is_main_process()):
            was_training = (model.module if distributed else model).training
            (model.module if distributed else model).eval()
            with torch.no_grad():
                val_result = run_test(cfg, model, local_rank, distributed)
            if was_training:
                (model.module if distributed else model).train()

            val_score = _extract_val_score(val_result)
            if val_score is None:
                logger.warning("[eval] val_score is None; run_test returned: %s", str(val_result)[:200])

            if (val_score is not None) and output_dir and _is_main_process():
                if val_score > best_score:
                    best_score = val_score
                    best_iter  = iteration
                    state = (model.module if distributed else model).state_dict()
                    torch.save({"iteration": iteration, "model": state, "score": best_score}, best_path)
                    logger.info(f"[ckpt] New BEST at iter {iteration}: {best_score:.4f} -> {best_path}")

        if iteration == max_iters or iteration == cfg.SOLVER.STOP_ITER:
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=int(total_training_time)))
    logging.getLogger("FADA.trainer").info(
        f"Total training time: {total_time_str} "
        f"({total_training_time / max_iters:.4f} s / it) "
        f"best: {best_score:.4f}@{best_iter}"
    )

    # === 训练结束保存 last ===
    if _is_main_process() and output_dir:
        state = (model.module if distributed else model).state_dict()
        torch.save(
            {"iteration": iteration, "model": state, "best_iter": best_iter, "best_score": best_score},
            last_path
        )
        logger.info(f"[ckpt] Saved LAST to {last_path}. Best {best_score:.4f} @ iter {best_iter}.")

    return model


# ===================== 测试 =====================
def run_test(cfg, model, local_rank, distributed):
    logger = logging.getLogger("FADA.tester")
    if local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing (DINOv2+Adapter) >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter(); data_time = AverageMeter()
    intersection_meter = AverageMeter(); union_meter = AverageMeter(); target_meter = AverageMeter()

    if distributed:
        model_eval = model.module
    else:
        model_eval = model
    torch.cuda.empty_cache()

    dataset_name = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name); mkdir(output_folder)

    test_data = build_dataset(cfg, mode='test', is_source=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data) if distributed else None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                                              num_workers=4, pin_memory=True, sampler=test_sampler)

    model_eval.eval()
    end = time.time()
    with torch.no_grad():
        for i, (x, y, _) in enumerate(test_loader):
            data_time.update(time.time() - end)
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True).long()

            # 前向（DINOv2SegPPM 已经上采样到原图大小）
            logits = model_eval(x)['logits']
            pred = logits
            # 如需保险，可强制对齐尺寸：
            if pred.shape[-2:] != y.shape[-2:]:
                pred = F.interpolate(pred, size=y.shape[-2:], mode='bilinear', align_corners=False)

            output = pred.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
            if distributed:
                torch.distributed.all_reduce(intersection); torch.distributed.all_reduce(union); torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection); union_meter.update(union); target_meter.update(target)

            batch_time.update(time.time() - end); end = time.time()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class); mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(cfg.MODEL.NUM_CLASSES):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))

    return {
        "mIoU": float(mIoU),
        "mAcc": float(mAcc),
        "allAcc": float(allAcc),
        "iou_class": iou_class.tolist(),
        "acc_class": accuracy_class.tolist(),
    }


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training (DINOv2+Adapter)")
    parser.add_argument("-cfg","--config-file", default="configs/segformer_mitbx_src.yaml", type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--skip-test", dest="skip_test", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--seed", type=int, default=8888)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    cfg.merge_from_file(args.config_file); cfg.merge_from_list(args.opts); cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir: mkdir(output_dir)
    logger = setup_logger("FADA", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus)); logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read(); logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)
    if not args.skip_test:
        run_test(cfg, model, args.local_rank, args.distributed)

if __name__ == "__main__":
    main()
