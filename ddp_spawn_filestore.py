"""
Windows 一键双卡并行测试（PyTorch 2.9.0+cu128 适配）
- 优先 DDP（Gloo + libuv + FileStore，自启动多进程）
- 自动尝试网卡别名；失败则回退 DataParallel
"""

import os, sys, time, tempfile, pathlib, traceback, socket
from datetime import timedelta

import torch
import torch.nn as nn
import torch.multiprocessing as mp

torch.set_num_threads(1)  # 避免小模型时抢占过多CPU

# ========= 小模型 + 随机数据 =========
class RandomImageDataset(torch.utils.data.Dataset):
    def __init__(self, n=4096, num_classes=10, shape=(3, 32, 32), seed=2025):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn((n, *shape), generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.y[i]

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x): return self.net(x)

# ========= DDP 子进程 =========
def _ddp_worker(rank: int, world: int, url: str, steps: int):
    # 这里默认走 libuv（Windows 稳定版应包含 libuv）
    os.environ["GLOO_DEVICE_TRANSPORT"] = "uv"

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler

    dist.init_process_group(
        backend="gloo",
        init_method=url,         # FileStore，无需 torchrun
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=180),
    )

    assert torch.cuda.is_available(), "未检测到 CUDA 设备"
    torch.cuda.set_device(rank)  # 0/1 进程各绑一张卡
    device = torch.device(f"cuda:{rank}")

    # 通信自检
    t = torch.tensor([rank], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"[DDP] 通信自检：sum(ranks) = {int(t.item())}（双卡应为 1）", flush=True)

    # 数据/采样/加载
    ds = RandomImageDataset()
    sp = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    dl = DataLoader(ds, batch_size=128, sampler=sp, num_workers=0, pin_memory=True, drop_last=True)

    # 模型/优化
    model = TinyCNN().to(device)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9)

    it = iter(dl)
    for step in range(20):
        try:
            x, y = next(it)
        except StopIteration:
            sp.set_epoch(step); it = iter(dl); x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = crit(ddp_model(x), y)
        loss.backward()
        opt.step()

        if step % 5 == 0:
            lt = torch.tensor([loss.item()], device=device)
            gather = [torch.zeros_like(lt) for _ in range(world)]
            dist.all_gather(gather, lt)
            if rank == 0:
                vals = [round(g.item(), 4) for g in gather]
                print(f"[DDP][step {step:02d}] loss@ranks = {vals}", flush=True)

    with torch.no_grad():
        pnorm = sum(p.norm(2).item() for p in ddp_model.parameters())
        pt = torch.tensor([pnorm], device=device)
        gathered = [torch.zeros_like(pt) for _ in range(world)]
        dist.all_gather(gathered, pt)
        if rank == 0:
            norms = [round(t.item(), 6) for t in gathered]
            print(f"[DDP] 参数范数：{norms}（应当非常接近）", flush=True)

    dist.barrier()
    dist.destroy_process_group()

# ========= 尝试 DDP（逐个网卡别名重试） =========
def try_run_ddp(world: int = 2) -> bool:
    if torch.cuda.device_count() < world:
        print(f"[DDP] 检测到 GPU 数 {torch.cuda.device_count()} < {world}，跳过 DDP。")
        return False

    # 候选网卡别名：优先自动探测（若 psutil 可用），再尝试常见别名
    candidates = [None]  # None 表示不显式指定，交给 Gloo 自选
    try:
        import psutil  # 可选依赖；若不存在自动忽略
        names = []
        stats = psutil.net_if_stats()
        for name, st in stats.items():
            if st.isup and "loopback" not in name.lower():
                names.append(name)
        # 优先把 "Ethernet"/"Wi-Fi" 放前面
        names_sorted = sorted(names, key=lambda n: (0 if n.lower().startswith("ethernet") else (1 if "wi" in n.lower() else 2)))
        candidates.extend(names_sorted)
    except Exception:
        pass
    # 常见别名兜底
    for alias in ("Ethernet", "Wi-Fi", "WLAN"):
        if alias not in candidates:
            candidates.append(alias)

    print("[DDP] 将按以下网卡别名尝试（None 表示不指定）：", candidates, flush=True)

    for ifname in candidates:
        # 每次尝试前设置/清理 env 变量（父进程设置对子进程生效）
        if ifname is None:
            os.environ.pop("GLOO_SOCKET_IFNAME", None)
            print("[DDP] 尝试：不指定 GLOO_SOCKET_IFNAME", flush=True)
        else:
            os.environ["GLOO_SOCKET_IFNAME"] = ifname
            print(f"[DDP] 尝试：GLOO_SOCKET_IFNAME={ifname}", flush=True)

        # FileStore：每次单独的临时文件
        fd, tmp = tempfile.mkstemp(); os.close(fd)
        url = "file:///" + pathlib.Path(tmp).as_posix()

        print("[DDP] 以 Gloo+libuv+FileStore 自启动多进程...", flush=True)
        print(f"[DDP] GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}", flush=True)

        ok = True
        try:
            mp.spawn(_ddp_worker, args=(world, url, 20), nprocs=world, join=True)
        except Exception:
            ok = False
            print(f"[DDP] 使用 ifname={ifname} 失败，异常：", flush=True)
            traceback.print_exc()
        finally:
            try: pathlib.Path(tmp).unlink(missing_ok=True)
            except Exception: pass

        if ok:
            print(f"[DDP] 成功：使用 ifname={ifname}", flush=True)
            return True

    print("[DDP] 所有尝试均失败。将回退到 DataParallel。", flush=True)
    return False

# ========= DataParallel 兜底 =========
def run_dataparallel():
    assert torch.cuda.is_available(), "未检测到 CUDA 设备"
    n = torch.cuda.device_count()
    assert n >= 2, f"需要 >= 2 张 GPU，当前 {n}"

    print(f"[DP] 使用 nn.DataParallel 在 {n} 张 GPU 上并行（兜底验证）", flush=True)
    device = torch.device("cuda:0")

    model = TinyCNN().to(device)
    model = nn.DataParallel(model, device_ids=list(range(n)), output_device=0)

    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    ds = RandomImageDataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    model.train()
    it = iter(dl)
    for step in range(20):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"[DP][step {step:02d}] loss = {loss.item():.4f}", flush=True)

    print("[DP] 运行结束（验证两卡并行计算，但不覆盖 DDP 通信同步）。", flush=True)

# ========= 主入口 =========
if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__} | CUDA 可用: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}", flush=True)

    t0 = time.time()
    ddp_ok = try_run_ddp(world=2)
    if not ddp_ok:
        run_dataparallel()
    print(f"总耗时：{time.time() - t0:.1f}s", flush=True)

    print("\n说明：\n"
          "- 若看到 [DDP] 打头的日志并成功结束，说明 **Windows 上 DDP 已跑通**；\n"
          "- 若自动回退到 [DP]，说明你这台机器上的 Gloo/libuv 仍未选对设备或构建不含 libuv；\n"
          "  这时可：1) 在上方 candidates 里固定成实际网卡名；2) 通过 `pip install psutil` 让脚本更好地自动探测；\n"
          "  或 3) 使用 WSL2/Linux + NCCL。", flush=True)
