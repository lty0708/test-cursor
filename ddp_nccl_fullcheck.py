# ddp_nccl_stress.py —— 点运行即可（WSL2 + NCCL）: 占显存 / 长时间 / 通信吞吐全检（强化版）
import os, sys, time, socket, traceback
from datetime import timedelta
import argparse

# ========= NCCL 环境（import torch 前设置，仅当前进程及其子进程） =========
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("NCCL_IB_DISABLE", "1")   # 单机一般无 IB
os.environ.setdefault("NCCL_P2P_DISABLE", "0")
os.environ.setdefault("NCCL_SHM_DISABLE", "0")
# 如遇网口选择问题：os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
# 可用来固定算法/协议（也可通过命令行参数覆盖）：
# os.environ.setdefault("NCCL_ALGO", "Ring")     # Ring / Tree
# os.environ.setdefault("NCCL_PROTO", "LL128")   # Simple / LL / LL128

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# ------------------------- 小工具 -------------------------
def seed_all(base=2025, add_rank=True):
    r = dist.get_rank() if (dist.is_initialized() and add_rank) else 0
    s = base + r
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def human(nbytes):
    for u in ["B","KB","MB","GB","TB"]:
        if nbytes < 1024 or u=="TB": return f"{nbytes:.1f}{u}"
        nbytes /= 1024

def barrier_print(*msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*msg, flush=True)

# ------------------------- 数据与模型 -------------------------
class RandomImageDataset(Dataset):
    def __init__(self, n=32768, num_classes=1000, shape=(3, 224, 224), seed=2025):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn((n, *shape), generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.y[i]

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, k, stride=1, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class BigConvNet(nn.Module):
    """较大卷积网络，可通过 --width/--depth 调整显存占用"""
    def __init__(self, num_classes=1000, width=128, depth=8):
        super().__init__()
        layers = []
        c = 3
        for i in range(depth):
            layers += [ConvBlock(c, width, k=3, s=1 if i%2==0 else 2)]
            c = width
            width *= 2 if (i%2==1) else 1
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

# ------------------------- 占显存工具 -------------------------
def fill_vram_to_frac(device, target_frac=0.6, safety_mb=1024):
    """尽量把显存使用提升到 target_frac；返回占位 tensor 列表（保持引用）。"""
    if target_frac <= 0 or target_frac > 0.98:
        return []
    free, total = torch.cuda.mem_get_info(device)
    target_used = int(total * target_frac)
    current_used = total - free
    need = target_used - current_used - safety_mb * 1024 * 1024
    if need <= 0:
        return []
    chunks, remain = [], need
    max_chunk = 512 * 1024 * 1024  # 512MB
    while remain > 0:
        this = min(remain, max_chunk)
        elems = this // 4  # float32
        if elems <= 0: break
        try:
            chunks.append(torch.empty(int(elems), dtype=torch.float32, device=device))
            remain -= int(elems) * 4
        except RuntimeError:
            max_chunk = max(16 * 1024 * 1024, max_chunk // 2)
            if max_chunk <= 16 * 1024 * 1024:
                break
    return chunks

# ------------------------- 基础通信自检 -------------------------
@torch.no_grad()
def check_collectives(device, warmup=2, iters=5):
    world = dist.get_world_size()
    rank = dist.get_rank()
    results = {}

    # 预热
    for _ in range(warmup):
        t = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(t)
        dist.barrier()

    # all_reduce 计时
    torch.cuda.synchronize(device); t0 = time.time()
    for _ in range(iters):
        tmp = torch.tensor([rank + 1.0], device=device)
        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    results["all_reduce(ms)"] = round((time.time() - t0) * 1000 / iters, 3)

    # 正确性
    val_chk = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(val_chk, op=dist.ReduceOp.SUM)
    expected = world * (world + 1) / 2
    results["all_reduce_ok"] = float(val_chk.item()) == expected

    # broadcast
    src = 0
    obj = torch.tensor([42.0 if rank == src else 0.0], device=device)
    torch.cuda.synchronize(device); t0 = time.time()
    for _ in range(iters): dist.broadcast(obj, src=src)
    torch.cuda.synchronize(device)
    results["broadcast(ms)"] = round((time.time() - t0) * 1000 / iters, 3)
    results["broadcast_ok"] = float(obj.item()) == 42.0

    # all_gather
    my = torch.tensor([rank], device=device, dtype=torch.int32)
    outs = [torch.zeros_like(my) for _ in range(world)]
    torch.cuda.synchronize(device); t0 = time.time()
    for _ in range(iters): dist.all_gather(outs, my)
    torch.cuda.synchronize(device)
    results["all_gather(ms)"] = round((time.time() - t0) * 1000 / iters, 3)
    results["all_gather_ok"] = [int(x.item()) for x in outs] == list(range(world))

    # barrier
    torch.cuda.synchronize(device); t0 = time.time()
    for _ in range(iters): dist.barrier()
    torch.cuda.synchronize(device)
    results["barrier(ms)"] = round((time.time() - t0) * 1000 / iters, 3)

    if rank == 0:
        print("[collectives]", results, flush=True)

# ------------------------- 强化通信压力：多算子 & 多消息大小 Sweep -------------------------
@torch.no_grad()
def _gather_scalars(val: float, device):
    t = torch.tensor([val], device=device, dtype=torch.float32)
    outs = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(outs, t)
    return [o.item() for o in outs]

def _iters_for_size(size_mb, iters_small, iters_large, threshold_mb):
    return iters_small if size_mb <= threshold_mb else iters_large

@torch.no_grad()
def comm_sweep(device, sizes_mb, iters_small=200, iters_large=40, threshold_mb=64, do_all_to_all=True):
    """
    对多种 collective 在多种消息大小下做吞吐测试，给出 per-GPU 近似 GB/s 的 min/avg。
    sizes_mb: list[int]，每个元素是“每卡参与该 collective 的负载大小（MB）”。
    """
    world = dist.get_world_size()
    rank = dist.get_rank()

    def report(op, size_mb, gbps_local):
        vals = _gather_scalars(gbps_local, device)
        if rank == 0:
            mn = round(min(vals), 2)
            av = round(sum(vals)/len(vals), 2)
            print(f"[comm_sweep] {op:<14} size={size_mb:>5}MB  per-GPU≈GB/s  min={mn:>6}  avg={av:>6}", flush=True)

    for size_mb in sizes_mb:
        iters = _iters_for_size(size_mb, iters_small, iters_large, threshold_mb)
        numel = (size_mb * 1024 * 1024) // 4  # float32

        # ---- all_reduce ----
        buf = torch.randn(int(numel), device=device, dtype=torch.float32)
        # 预热
        for _ in range(5): dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device); t0 = time.time()
        for _ in range(iters): dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        dt = (time.time() - t0) / max(iters, 1)
        bytes_one = size_mb * 1024 * 1024
        xfer_per_gpu = 2.0 * (world - 1) / world * bytes_one  # ring 模型
        gbps_local = (xfer_per_gpu / dt) / (1024**3)
        report("all_reduce", size_mb, gbps_local)

        # ---- all_gather ----
        inp = torch.randn(int(numel), device=device, dtype=torch.float32)
        outs = [torch.empty_like(inp) for _ in range(world)]
        for _ in range(5): dist.all_gather(outs, inp)
        torch.cuda.synchronize(device); t0 = time.time()
        for _ in range(iters): dist.all_gather(outs, inp)
        torch.cuda.synchronize(device)
        dt = (time.time() - t0) / max(iters, 1)
        # 每卡传输量 ~ size_mb*(N-1) MB（发送 & 接收各 size_mb*(N-1)）
        xfer_per_gpu = (size_mb * (world - 1)) * 1024 * 1024
        gbps_local = (xfer_per_gpu / dt) / (1024**3)
        report("all_gather", size_mb, gbps_local)

        # ---- reduce_scatter ----
        # 输入需要 world 倍长度，输出每卡 size_mb
        inp = torch.randn(int(numel * world), device=device, dtype=torch.float32)
        out = torch.empty(int(numel), device=device, dtype=torch.float32)
        for _ in range(5): dist.reduce_scatter(out, list(inp.chunk(world)), op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device); t0 = time.time()
        for _ in range(iters): dist.reduce_scatter(out, list(inp.chunk(world)), op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        dt = (time.time() - t0) / max(iters, 1)
        # 每卡传输量 ~ size_mb*(N-1) MB
        xfer_per_gpu = (size_mb * (world - 1)) * 1024 * 1024
        gbps_local = (xfer_per_gpu / dt) / (1024**3)
        report("reduce_scatter", size_mb, gbps_local)

        # ---- all_to_all （可选）----
        if do_all_to_all:
            # 每卡发送 size_mb，总共平均分给 N 个卡
            # 形状用 [world, numel_per_peer]，便于 equal split
            numel_per_peer = int(numel // world)
            if numel_per_peer == 0:
                continue
            inp = torch.randn(world * numel_per_peer, device=device, dtype=torch.float32)
            out = torch.empty_like(inp)
            for _ in range(5): dist.all_to_all_single(out, inp, [], [])
            torch.cuda.synchronize(device); t0 = time.time()
            for _ in range(iters): dist.all_to_all_single(out, inp, [], [])
            torch.cuda.synchronize(device)
            dt = (time.time() - t0) / max(iters, 1)
            # 粗略估计：每卡发送 + 接收 ≈ 2 * size_mb
            xfer_per_gpu = (2.0 * size_mb) * 1024 * 1024
            gbps_local = (xfer_per_gpu / dt) / (1024**3)
            report("all_to_all", size_mb, gbps_local)

# ------------------------- 训练（worker 内执行） -------------------------
def ddp_worker(rank, world, dist_url, args):
    try:
        torch.set_num_threads(1)
        assert torch.cuda.is_available(), "CUDA 不可用；请确认在 WSL 解释器下运行且 nvidia-smi 正常"
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        # 可选：通过参数覆盖 NCCL 算法/协议
        if args.nccl_algo:  os.environ["NCCL_ALGO"]  = args.nccl_algo
        if args.nccl_proto: os.environ["NCCL_PROTO"] = args.nccl_proto

        # device_id=rank，去掉 barrier 警告
        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world,
            rank=rank,
            timeout=timedelta(seconds=600),
            device_id=rank,
        )

        if rank == 0:
            print(f"PyTorch: {torch.__version__}")
            try: print("NCCL:", torch.cuda.nccl.version())
            except Exception: pass
            print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
            torch.backends.cuda.matmul.allow_tf32 = True  # 需要更快可以开

        seed_all(args.seed, add_rank=True)
        torch.backends.cudnn.benchmark = True

        # —— 占显存（把使用量拉到 target_vram_frac）——
        fillers = []
        if args.target_vram_frac > 0:
            fillers = fill_vram_to_frac(device, args.target_vram_frac, args.safety_mb)
        free, total = torch.cuda.mem_get_info(device)
        used = total - free
        mem_local = torch.tensor([used, total], device=device, dtype=torch.int64)
        mem_list = [torch.zeros_like(mem_local) for _ in range(world)]
        dist.all_gather(mem_list, mem_local)
        if rank == 0:
            pretty = [(human(m[0].item()), human(m[1].item())) for m in mem_list]
            print(f"[memory] used/total per-rank: {pretty}", flush=True)

        # —— 基础自检 & 强化通信吞吐 sweep ——
        check_collectives(device)
        if args.comm_suite:
            sizes = [int(s) for s in args.comm_sizes_mb.split(",")]
            comm_sweep(
                device,
                sizes_mb=sizes,
                iters_small=args.comm_iters_small,
                iters_large=args.comm_iters_large,
                threshold_mb=args.comm_threshold_mb,
                do_all_to_all=not args.skip_all_to_all
            )

        # —— 数据 ——
        ds = RandomImageDataset(
            n=args.num_samples,
            shape=(3, args.image_size, args.image_size),
            num_classes=args.num_classes,
            seed=args.seed
        )
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        dl = DataLoader(
            ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.workers, pin_memory=True, drop_last=True,
            persistent_workers=args.workers > 0
        )

        # —— 模型/优化（较大网络，显存占用更高）——
        model = BigConvNet(num_classes=args.num_classes, width=args.width, depth=args.depth).to(device)
        ddp = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            gradient_as_bucket_view=True,
            bucket_cap_mb=args.bucket_cap_mb,     # 控制梯度桶大小，影响通信 pattern
            find_unused_parameters=False
        )
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(ddp.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

        # —— 训练 ——
        it = iter(dl)
        seen = 0
        t0 = time.time()
        for step in range(1, args.steps + 1):
            try:
                x, y = next(it)
            except StopIteration:
                sampler.set_epoch(step)
                it = iter(dl)
                x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=args.amp,
                                    dtype=torch.float16 if args.amp_dtype=="fp16" else torch.bfloat16):
                logits = ddp(x)
                loss = criterion(logits, y) / args.accum_steps

            scaler.scale(loss).backward()
            if step % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            seen += x.size(0)

            if step % args.log_every == 0:
                with torch.no_grad():
                    l = torch.tensor([loss.item() * args.accum_steps], device=device)
                    gathers = [torch.zeros_like(l) for _ in range(world)]
                    dist.all_gather(gathers, l)
                    elapsed = time.time() - t0
                    seen_t = torch.tensor([seen], device=device, dtype=torch.int64)
                    seen_list = [torch.zeros_like(seen_t) for _ in range(world)]
                    dist.all_gather(seen_list, seen_t)
                    if rank == 0:
                        losses = [round(g.item(), 4) for g in gathers]
                        total_samples = int(sum(x.item() for x in seen_list))
                        ips = total_samples / max(elapsed, 1e-6)
                        print(f"[step {step:04d}] loss@ranks={losses} | {ips:.1f} img/s | elapsed={elapsed:.1f}s", flush=True)

        # —— 一致性检查 ——
        with torch.no_grad():
            pn = sum(p.norm(2).item() for p in ddp.module.parameters())
            t = torch.tensor([pn], device=device)
            outs = [torch.zeros_like(t) for _ in range(world)]
            dist.all_gather(outs, t)
            if rank == 0:
                print("[check] param_norms=", [round(x.item(), 6) for x in outs], flush=True)

        # —— 仅 rank0 保存 ——
        if rank == 0:
            ckpt = {"state_dict": ddp.module.state_dict(), "optimizer": optimizer.state_dict(), "args": vars(args)}
            os.makedirs("ckpts", exist_ok=True)
            torch.save(ckpt, "ckpts/ddp_nccl_stress.pt")
            print("Saved ckpt -> ckpts/ddp_nccl_stress.pt", flush=True)

    except Exception as e:
        barrier_print("[worker error]", rank, ":", repr(e))
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

# ------------------------- Orchestrator -------------------------
def main():
    p = argparse.ArgumentParser("WSL2 + NCCL DDP 压力测试（显存/通信/长训练，强化版）")
    # 训练规模
    p.add_argument("--gpus", type=int, default=2, help="使用多少张 GPU")
    p.add_argument("--steps", type=int, default=200, help="总训练步数")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-samples", type=int, default=65536)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.15)
    p.add_argument("--accum-steps", type=int, default=1, help="梯度累积步数")
    p.add_argument("--amp", action="store_true", help="开启 AMP")
    p.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16","bf16"])
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=2025)
    # 模型规模
    p.add_argument("--width", type=int, default=128, help="模型初始通道（越大越占显存）")
    p.add_argument("--depth", type=int, default=8, help="网络层数（越大越占显存）")
    p.add_argument("--bucket-cap-mb", type=int, default=25, help="DDP 梯度桶大小（MB）")
    # 显存占用
    p.add_argument("--target-vram-frac", type=float, default=0.6, help="把显存使用量拉到该比例（0~0.98，0 关闭）")
    p.add_argument("--safety-mb", type=int, default=1024, help="预留显存（MB），防止 OOM")
    # 通信压力（基础）
    p.add_argument("--comm-suite", action="store_true", help="启用多算子多大小通信吞吐 sweep")
    p.add_argument("--comm-sizes-mb", type=str, default="4,16,64,256,1024", help="逗号分隔的每卡负载大小（MB）")
    p.add_argument("--comm-iters-small", type=int, default=200)
    p.add_argument("--comm-iters-large", type=int, default=40)
    p.add_argument("--comm-threshold-mb", type=int, default=64, help="大/小负载分界（MB）")
    p.add_argument("--skip-all-to-all", action="store_true", help="跳过 all_to_all 测试")
    # NCCL 算法/协议（可选）
    p.add_argument("--nccl-algo", type=str, default=None, choices=[None, "Ring", "Tree"])
    p.add_argument("--nccl-proto", type=str, default=None, choices=[None, "Simple", "LL", "LL128"])
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用。确保 PyCharm 解释器是 WSL 内的 Python 且 nvidia-smi 正常。")
        sys.exit(1)

    total = torch.cuda.device_count()
    world = min(args.gpus, total)
    if world < 1:
        print("❌ 未检测到可用 GPU。"); sys.exit(1)
    if world < args.gpus:
        print(f"⚠️ 仅检测到 {total} 张 GPU，将使用 {world} 张。")

    port = find_free_port()
    dist_url = f"tcp://127.0.0.1:{port}"
    print(f"[Launcher] Using world_size={world}, init_method={dist_url}")
    print("[Launcher] GPUs:", [torch.cuda.get_device_name(i) for i in range(total)], flush=True)

    mp.set_start_method("spawn", force=True)
    mp.spawn(ddp_worker, args=(world, dist_url, args), nprocs=world, join=True)

if __name__ == "__main__":
    if os.environ.get("RANK") is not None:
        # 兼容 torchrun
        class Dummy: pass
        d = Dummy()
        d.gpus = int(os.environ.get("WORLD_SIZE", "1"))
        d.steps = int(os.environ.get("STEPS", "200"))
        d.batch_size = int(os.environ.get("BATCH_SIZE", "96"))
        d.image_size = int(os.environ.get("IMAGE_SIZE", "224"))
        d.num_samples = int(os.environ.get("NUM_SAMPLES", "65536"))
        d.num_classes = int(os.environ.get("NUM_CLASSES", "1000"))
        d.workers = int(os.environ.get("WORKERS", "4"))
        d.lr = float(os.environ.get("LR", "0.15"))
        d.accum_steps = int(os.environ.get("ACCUM_STEPS", "1"))
        d.amp = os.environ.get("AMP", "0") == "1"
        d.amp_dtype = os.environ.get("AMP_DTYPE", "fp16")
        d.log_every = int(os.environ.get("LOG_EVERY", "25"))
        d.width = int(os.environ.get("WIDTH", "128"))
        d.depth = int(os.environ.get("DEPTH", "8"))
        d.bucket_cap_mb = int(os.environ.get("BUCKET_CAP_MB", "25"))
        d.target_vram_frac = float(os.environ.get("TARGET_VRAM_FRAC", "0.6"))
        d.safety_mb = int(os.environ.get("SAFETY_MB", "1024"))
        d.comm_suite = os.environ.get("COMM_SUITE", "0") == "1"
        d.comm_sizes_mb = os.environ.get("COMM_SIZES_MB", "4,16,64,256,1024")
        d.comm_iters_small = int(os.environ.get("COMM_ITERS_SMALL", "200"))
        d.comm_iters_large = int(os.environ.get("COMM_ITERS_LARGE", "40"))
        d.comm_threshold_mb = int(os.environ.get("COMM_THRESHOLD_MB", "64"))
        d.skip_all_to_all = os.environ.get("SKIP_ALL_TO_ALL", "0") == "1"
        d.nccl_algo = os.environ.get("NCCL_ALGO_ARG", None)
        d.nccl_proto = os.environ.get("NCCL_PROTO_ARG", None)
        rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
        ddp_worker(rank, world, "env://", d)
    else:
        main()

