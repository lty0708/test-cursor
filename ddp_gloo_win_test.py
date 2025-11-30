import os
# —— 这些 env 变量务必在 import torch 前设置（也可在外部 PowerShell 里设置）——
os.environ.setdefault("GLOO_DEVICE_TRANSPORT", "uv")   # Windows: Gloo 走 libuv
# 如需固定网卡，取消下一行注释并改成你的网卡别名（例如 "Ethernet" 或 "Wi-Fi"）
#os.environ.setdefault("GLOO_SOCKET_IFNAME", "Ethernet")

from datetime import timedelta
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# ------------ 合成数据集（随机） ------------
class RandomImageDataset(Dataset):
    def __init__(self, length=8192, num_classes=10, image_shape=(3, 32, 32), seed=1234):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn((length, *image_shape), generator=g)
        self.y = torch.randint(0, num_classes, (length,), generator=g)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        print(f"idx: {idx}")
        return self.x[idx], self.y[idx]


# ------------ 一个小 CNN 作为玩具模型 ------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    #1111
    def forward(self, x):
        return self.net(x)

def setup_dist():
    # torchrun 会注入以下环境变量
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    # Windows 必须用 gloo
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=timedelta(seconds=180)
    )

    # 绑卡
    assert torch.cuda.is_available(), "未检测到可用 CUDA，请确认驱动/环境。"
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device

def main():
    # 1) 基础环境
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    rank, world_size, local_rank, device = setup_dist()
    if torch.cuda.device_count() < world_size:
        if rank == 0:
            print(f"警告：本机 GPUs={torch.cuda.device_count()} < world_size={world_size}")
        dist.barrier()

    if rank == 0:
        print(f"[World {world_size}] GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    # 2) 快速通信自检（all-reduce）
    t = torch.tensor([rank], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"[通信自检] ranks 求和：{int(t.item())}（应为 0+1=2）")

    # 3) 数据/采样器/加载器
    dataset = RandomImageDataset(length=4096, num_classes=10)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset, batch_size=128, sampler=sampler,
        num_workers=0, pin_memory=True, drop_last=True
    )

    # 4) 模型/优化器
    model = TinyCNN(num_classes=10).to(device)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1, momentum=0.9)

    # 5) 训练若干 step，验证梯度同步
    ddp_model.train()
    steps = 50  # 足够看到 loss 下降和参数同步
    it = iter(loader)

    for step in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            sampler.set_epoch(step)  # 重设随机种子
            it = iter(loader)
            x, y = next(it)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = ddp_model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            # 收集各 rank 的 loss 看是否接近
            loss_tensor = torch.tensor([loss.item()], device=device)
            gather_list = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
            dist.all_gather(gather_list, loss_tensor)

            if rank == 0:
                losses = [round(t.item(), 4) for t in gather_list]
                print(f"[step {step:03d}] loss@ranks={losses}")

    # 6) 参数范数对齐性检查
    with torch.no_grad():
        p_norm = sum(p.norm(2).item() for p in ddp_model.parameters())
        p_norm = torch.tensor([p_norm], device=device)
        gathered = [torch.zeros_like(p_norm) for _ in range(world_size)]
        dist.all_gather(gathered, p_norm)
        if rank == 0:
            norms = [round(t.item(), 6) for t in gathered]
            print(f"[参数范数] ranks={norms}（应当非常接近）")

    # 7) 只在 rank0 存个检查点
    if rank == 0:
        torch.save(ddp_model.state_dict(), "ddp_gloo_win_test.pt")
        print("Saved checkpoint: ddp_gloo_win_test.pt")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    # Windows 默认 spawn，多进程下主入口必须保护
    main()
