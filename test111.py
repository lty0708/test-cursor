import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def banner(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def check_env():
    banner("Step 0: 环境 & 设备信息")
    print(f"PyTorch version           : {torch.__version__}")
    print(f"PyTorch built with CUDA   : {torch.version.cuda}")
    print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，请检查显卡驱动 / CUDA / PyTorch 安装。")
        return 0

    num_devices = torch.cuda.device_count()
    print(f"可见 GPU 数量              : {num_devices}")

    for i in range(num_devices):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        torch.cuda.set_device(i)
        free, total = torch.cuda.mem_get_info()
        print(f"  - GPU {i}: {name}, compute capability {cap}")
        print(f"      总显存: {total/1024**3:.2f} GB, 当前空闲: {free/1024**3:.2f} GB")

    return num_devices


# ---------------- 单卡测试用的小网络 ---------------- #

class SmallNet(nn.Module):
    """一个很简单的小网络，用来测试训练流程。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # [B, 32, 32, 32]
        x = F.max_pool2d(x, 2)         # [B, 32, 16, 16]
        x = F.relu(self.conv2(x))      # [B, 64, 16, 16]
        x = F.max_pool2d(x, 2)         # [B, 64, 8, 8]
        x = x.view(x.size(0), -1)      # [B, 64*8*8]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------- 每张卡单独测试的函数 ---------------- #

def test_basic_ops(gpu_idx: int):
    banner(f"[GPU {gpu_idx}] Step 1: 基础张量运算 / 大矩阵乘法")
    try:
        torch.cuda.set_device(gpu_idx)
        device = torch.device(f"cuda:{gpu_idx}")

        torch.cuda.synchronize()
        t0 = time.time()

        a = torch.randn(4096, 4096, device=device)
        b = torch.randn(4096, 4096, device=device)
        c = a @ b

        torch.cuda.synchronize()
        t1 = time.time()

        print("✅ 大矩阵乘法完成")
        print(f"结果均值 (防止 lazy) : {c.mean().item():.6f}")
        print(f"耗时                 : {t1 - t0:.3f} 秒")
        return True
    except Exception as e:
        print("❌ 基础运算测试失败:", e)
        return False


def test_conv_and_backward(gpu_idx: int):
    banner(f"[GPU {gpu_idx}] Step 2: cuDNN 卷积 + 反向传播")
    try:
        torch.cuda.set_device(gpu_idx)
        device = torch.device(f"cuda:{gpu_idx}")

        model = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(device)
        x = torch.randn(32, 64, 128, 128, device=device, requires_grad=True)

        torch.cuda.synchronize()
        t0 = time.time()

        y = model(x)
        loss = y.mean()
        loss.backward()

        torch.cuda.synchronize()
        t1 = time.time()

        print("✅ 卷积前向 + 反向 正常完成")
        print(f"loss                : {loss.item():.6f}")
        print(f"耗时                 : {t1 - t0:.3f} 秒")
        return True
    except Exception as e:
        print("❌ 卷积 / 反向传播 测试失败:", e)
        return False


def test_training_loop(gpu_idx: int):
    banner(f"[GPU {gpu_idx}] Step 3: 小模型训练若干步 (FP32)")
    try:
        torch.cuda.set_device(gpu_idx)
        device = torch.device(f"cuda:{gpu_idx}")

        model = SmallNet().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        steps = 20
        for step in range(steps):
            inputs = torch.randn(64, 3, 32, 32, device=device)
            targets = torch.randint(0, 10, (64,), device=device)

            optim.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()

            if (step + 1) % 5 == 0:
                print(f"  Step {step+1}/{steps}  loss={loss.item():.4f}")

        print("✅ FP32 训练流程正常")
        return True
    except Exception as e:
        print("❌ FP32 训练测试失败:", e)
        return False


def test_amp_mixed_precision(gpu_idx: int):
    banner(f"[GPU {gpu_idx}] Step 4: AMP 混合精度 / Tensor Core")
    try:
        from torch.cuda.amp import autocast, GradScaler

        torch.cuda.set_device(gpu_idx)
        device = torch.device(f"cuda:{gpu_idx}")

        model = SmallNet().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

        model.train()
        steps = 20
        for step in range(steps):
            inputs = torch.randn(64, 3, 32, 32, device=device)
            targets = torch.randint(0, 10, (64,), device=device)

            optim.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if (step + 1) % 5 == 0:
                print(f"  AMP Step {step+1}/{steps}  loss={loss.item():.4f}")

        print("✅ AMP 混合精度训练流程正常")
        return True
    except Exception as e:
        print("❌ AMP 测试失败:", e)
        return False


def test_memory_stress(gpu_idx: int):
    banner(f"[GPU {gpu_idx}] Step 5: 简单显存压力测试")
    try:
        torch.cuda.set_device(gpu_idx)
        # 当前设备的空闲显存
        free, total = torch.cuda.mem_get_info()
        target_bytes = int(free * 0.3)  # 占用大约 30% 空闲显存
        elem_count = target_bytes // 4  # float32 = 4 bytes

        print(f"当前空闲显存约        : {free/1024**3:.2f} GB")
        print(f"计划分配张量大小约    : {target_bytes/1024**3:.2f} GB")

        big_tensor = torch.empty(elem_count, dtype=torch.float32,
                                 device=f"cuda:{gpu_idx}")
        big_tensor.normal_()

        print("✅ 显存大块分配成功，没有 OOM")
        del big_tensor
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        print("⚠️ 显存压力测试触发 RuntimeError (可能是 OOM，属正常保护)：", e)
        return False
    except Exception as e:
        print("❌ 显存压力测试出现异常:", e)
        return False


# ---------------- 多卡联合测试 ---------------- #

def test_multi_gpu_comm(gpu_indices):
    """测试两张卡之间的拷贝 + DataParallel 前向/反向"""
    if len(gpu_indices) < 2:
        return {"cross_copy": False, "dataparallel": False}

    gpu0, gpu1 = gpu_indices[0], gpu_indices[1]
    results = {"cross_copy": False, "dataparallel": False}

    banner(f"[GPU {gpu0} & GPU {gpu1}] Step 6: 多 GPU 跨卡拷贝 / DataParallel")

    # 1) 跨卡张量拷贝测试
    try:
        torch.cuda.set_device(gpu0)
        x = torch.randn(2048, 2048, device=f"cuda:{gpu0}")
        y = x.to(f"cuda:{gpu1}")
        z = y.to(f"cuda:{gpu0}")
        diff = (x - z).abs().max().item()
        print(f"跨卡拷贝 max|x - z| = {diff:e}")
        if diff < 1e-5:
            print("✅ 跨卡拷贝数据一致")
            results["cross_copy"] = True
        else:
            print("❌ 跨卡拷贝数据不一致（数值误差过大）")
    except Exception as e:
        print("❌ 跨卡拷贝测试失败:", e)

    # 2) DataParallel 测试（两张卡一起算）
    try:
        torch.cuda.set_device(gpu0)
        device0 = torch.device(f"cuda:{gpu0}")

        model = SmallNet().to(device0)
        dp_model = nn.DataParallel(model, device_ids=[gpu0, gpu1])
        optim = torch.optim.SGD(dp_model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        dp_model.train()

        inputs = torch.randn(128, 3, 32, 32, device=device0)
        targets = torch.randint(0, 10, (128,), device=device0)

        optim.zero_grad(set_to_none=True)
        outputs = dp_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optim.step()

        print("✅ DataParallel 前向 + 反向 正常完成")
        print(f"DataParallel loss   : {loss.item():.4f}")
        results["dataparallel"] = True
    except Exception as e:
        print("❌ DataParallel 测试失败:", e)

    return results


# ---------------- main ---------------- #

def main():
    num_devices = check_env()
    if num_devices == 0:
        return

    all_results = {}

    # 对每一张卡单独测试
    for gpu_idx in range(num_devices):
        name = torch.cuda.get_device_name(gpu_idx)
        print("\n" + "#" * 80)
        print(f"开始测试 GPU {gpu_idx}: {name}")
        print("#" * 80)

        results = {}
        results["basic_ops"] = test_basic_ops(gpu_idx)
        results["conv_backward"] = test_conv_and_backward(gpu_idx)
        results["training_fp32"] = test_training_loop(gpu_idx)
        results["amp"] = test_amp_mixed_precision(gpu_idx)
        results["memory_stress"] = test_memory_stress(gpu_idx)
        all_results[gpu_idx] = results
        print(results)

    # 多卡联合测试（如果有 2 张及以上）
    multi_gpu_results = {}
    if num_devices >= 2:
        multi_gpu_results = test_multi_gpu_comm(list(range(num_devices)))

    # 总结
    banner("总 结")

    for gpu_idx, res in all_results.items():
        name = torch.cuda.get_device_name(gpu_idx)
        print(f"\nGPU {gpu_idx}: {name}")
        for k, v in res.items():
            print(f"  {k:15s} : {'OK' if v else 'FAILED'}")

    if multi_gpu_results:
        print("\n多 GPU 联合测试:")
        for k, v in multi_gpu_results.items():
            print(f"  {k:15s} : {'OK' if v else 'FAILED'}")

    print("\n说明：")
    print("- 单卡所有项目都 OK，说明该卡在 PyTorch 下基础计算路径是健康的；")
    print("- 多 GPU 测试 OK，说明两张卡之间拷贝、DataParallel 正常；")
    print("- 如果某项 FAILED，把终端的报错信息复制出来，我帮你一起排查。")


if __name__ == "__main__":
    main()
