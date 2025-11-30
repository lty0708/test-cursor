# -*- coding: utf-8 -*-
# ddp_win_final_probe.py —— Windows 双卡 DDP 终极自检：libuv / tcp 两套传输 + 网卡穷举 + 自动回退 DP
import os, sys, time, tempfile, pathlib, subprocess, traceback
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

STEPS = 12
torch.set_num_threads(1)

def log(*a): print(*a, flush=True)

# ---------------- 工具：枚举网卡别名（尽量无依赖） ----------------
def detect_ifnames():
    names = []

    # 1) PowerShell: Up 的 NetAdapter
    try:
        out = subprocess.check_output(
            ['powershell', '-NoProfile',
             "Get-NetAdapter | Where-Object {$_.Status -eq 'Up'} | Select-Object -ExpandProperty Name"],
            text=True, encoding='utf-8', errors='ignore')
        names += [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        pass

    # 2) PowerShell: Connected 的 IPInterface
    try:
        out = subprocess.check_output(
            ['powershell', '-NoProfile',
             "Get-NetIPInterface | Where-Object {$_.ConnectionState -eq 'Connected'} | Select-Object -ExpandProperty InterfaceAlias"],
            text=True, encoding='utf-8', errors='ignore')
        names += [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        pass

    # 3) netsh（最后一列是名字）
    try:
        out = subprocess.check_output(['netsh', 'interface', 'ipv4', 'show', 'interfaces'],
                                      text=True, encoding='utf-8', errors='ignore')
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith('Idx') or line.startswith('---'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                names.append(" ".join(parts[4:]).strip())
    except Exception:
        pass

    # 4) ipconfig /all（“适配器 XXX:” 或 “adapter XXX:”）
    try:
        out = subprocess.check_output(['ipconfig', '/all'], text=True, encoding='utf-8', errors='ignore')
        for line in out.splitlines():
            t = line.strip()
            if '适配器' in t and t.endswith(':'):
                names.append(t.split('适配器', 1)[-1].rstrip(':').strip())
            elif 'adapter' in t and t.endswith(':'):
                names.append(t.split('adapter', 1)[-1].rstrip(':').strip())
    except Exception:
        pass

    # 常见兜底
    names += ["以太网", "以太网 2", "Ethernet", "Wi-Fi", "WLAN", "蓝牙网络连接",
              "vEthernet (Default Switch)", "vEthernet (WSL)"]

    # 去重保序 + 过滤明显无效项
    seen, uniq = set(), []
    for n in names:
        if not n:
            continue
        # 过滤 Loopback/伪接口（Gloo 常不支持）
        low = n.lower()
        if "loopback" in low or "pseudo" in low:
            continue
        if n not in seen:
            seen.add(n); uniq.append(n)
    return uniq

# ---------------- DDP 训练体 ----------------
def ddp_worker(rank, world, url, steps):
    if torch.cuda.is_available() and torch.cuda.device_count() >= world:
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    dist.init_process_group(backend="gloo", init_method=url,
                            world_size=world, rank=rank,
                            timeout=timedelta(seconds=180))

    # 通信自检
    t = torch.tensor([rank + 1], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if rank == 0:
        log(f"[DDP] all_reduce 校验 (应为 {world*(world+1)//2}) -> {int(t.item())}")

    # 简单线性模型
    model = nn.Linear(1024, 10, bias=False).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index] if device.type=="cuda" else None
    )
    opt = optim.SGD(ddp_model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(steps):
        x = torch.randn(64, 1024, device=device)
        y = torch.randint(0, 10, (64,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(ddp_model(x), y)
        loss.backward(); opt.step()
        if rank == 0 and step % 4 == 0:
            log(f"[DDP][step {step:02d}] loss={loss.item():.4f}")

    dist.destroy_process_group()

# ---------------- 单次尝试：传输方式 + ifname ----------------
def try_one(transport: str, ifname: str | None) -> bool:
    # 先清理
    for k in ("GLOO_DEVICE_TRANSPORT", "GLOO_SOCKET_IFNAME", "USE_LIBUV"):
        os.environ.pop(k, None)

    if transport == "uv":
        os.environ["GLOO_DEVICE_TRANSPORT"] = "uv"
        os.environ["USE_LIBUV"] = "1"
    else:
        os.environ["GLOO_DEVICE_TRANSPORT"] = "tcp"
        os.environ["USE_LIBUV"] = "0"

    if ifname:
        os.environ["GLOO_SOCKET_IFNAME"] = ifname

    # FileStore
    fd, tmp = tempfile.mkstemp(prefix='gloo_store_', suffix='.init')
    os.close(fd)
    url = pathlib.Path(tmp).as_uri()

    log(f"[DDP] 尝试: transport={transport}, ifname={ifname}")
    ok = True
    try:
        mp.set_start_method("spawn", force=True)
        world = min(2, max(1, torch.cuda.device_count()))
        mp.spawn(ddp_worker, args=(world, url, STEPS), nprocs=world, join=True)
    except Exception:
        ok = False
        traceback.print_exc()
    finally:
        try: pathlib.Path(tmp).unlink(missing_ok=True)
        except Exception: pass
    return ok

# ---------------- DP 兜底 ----------------
def run_dp():
    log("[DP] 回退：nn.DataParallel 并行（验证两卡 CUDA 正常）")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(nn.Linear(1024, 10, bias=False)).to(device)
    opt = optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    for step in range(STEPS):
        x = torch.randn(128, 1024, device=device)
        y = torch.randint(0, 10, (128,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward(); opt.step()
        if step % 4 == 0:
            log(f"[DP][step {step:02d}] loss={loss.item():.4f}")
    log("[DP] 完成。")

# ---------------- 主流程 ----------------
if __name__ == "__main__":
    log(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        log("[GPU]", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

    # 先尝试 libuv（Windows 推荐/常用），再尝试 tcp（禁用 libuv）
    tried = []
    ifnames = [None] + detect_ifnames()
    transports = ["uv", "tcp"]  # 顺序很重要

    success = False
    for t in transports:
        log(f"\n========== 尝试传输: {t} ==========")
        for n in ifnames:
            tag = (t, n)
            if tag in tried:  # 防重
                continue
            tried.append(tag)
            ok = try_one(t, n)
            if ok:
                log(f"\n✅ 成功：transport={t}, ifname={n}\n")
                success = True
                break
        if success:
            break

    if not success:
        log("\n❌ 仍未跑通 DDP，进入 DP 兜底…\n")
        run_dp()
        log("""
[诊断与建议]
1) 你这台 Windows 的 Gloo 无法为任何网卡创建设备（unsupported gloo device）。
   常见触发：只有 IPv6、VPN/虚拟网卡占优、以太网禁用 IPv4、网卡名含特殊字符、杀软/防火墙拦截、驱动/系统异常。
2) 立刻可做的本机检查（任意一条可能就能修好）：
   - 以管理员身份运行本脚本 / IDE。
   - 暂时关闭 VPN/代理/虚拟网卡（如 VMware/Hyper-V 等），只保留一个可用的以太网或 Wi-Fi。
   - 打开“网络适配器”属性，确保启用了“Internet 协议版本 4 (TCP/IPv4)”并分配了 IPv4。
   - Windows 防火墙里为 Python 允许专用网络通信；或临时关闭防火墙做对比测试。
   - 尝试只保留一个活动网卡并重命名为简单英文（如 “Ethernet”），重启后再跑脚本。
3) 若必须稳定用 DDP：
   - 最稳路线：WSL2 / 原生 Linux + NCCL（torchrun --nproc-per-node=2，后端 nccl）。
   - 或在 Windows 源码编译带 libuv 的 PyTorch/Gloo（官方二进制在个别系统上存在回归的已知问题）。
""")
