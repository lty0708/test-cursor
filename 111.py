# cudnn_probe.py — 一键体检 cuDNN（PyTorch 单机单卡微基准）
# 直接 python cudnn_probe.py 运行；可用参数见 parse_args()

import os, time, argparse, math, traceback
import torch
import torch.nn as nn
import torch.optim as optim

# —— 建议性的默认（可被命令行覆盖）——
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # 真实训练里推荐 True；基准里我们会切换对比

def nice_bool(x: bool) -> str: return "Yes" if x else "No"

def env_info():
    print("=== Environment ===")
    print(f"PyTorch            : {torch.__version__}")
    print(f"CUDA available     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device        : {torch.cuda.get_device_name(0)}")
        print(f"CC (SM)            : {torch.cuda.get_device_capability(0)}")
        try:
            tot = torch.cuda.get_device_properties(0).total_memory
            print(f"VRAM total         : {tot/1024**3:.1f} GB")
        except Exception:
            pass
    # cuDNN 基本信息
    try:
        v = torch.backends.cudnn.version()
    except Exception:
        v = None
    print(f"cuDNN available    : {nice_bool(torch.backends.cudnn.is_available())}")
    print(f"cuDNN version      : {v if v is not None else 'Unknown'}")
    print(f"cudnn.enabled      : {nice_bool(torch.backends.cudnn.enabled)}")
    print(f"cudnn.benchmark    : {nice_bool(torch.backends.cudnn.benchmark)}")
    # AMP/TF32 支持情况
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"AMP bf16 supported : {nice_bool(bf16_ok)}")
    print(f"AMP fp16 supported : {nice_bool(torch.cuda.is_available())}")
    print(f"allow_tf32         : {nice_bool(getattr(torch.backends.cuda.matmul, 'allow_tf32', False))}")
    print()

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

@torch.no_grad()
def measure_one_step(forward_fn, backward=False):
    # 仅计时前向或前后向（根据 backward）。backward=False 时不跟踪梯度。
    start = time.time()
    y = forward_fn()
    if backward:
        # 这里 backward 需要在外部以 no_grad 关闭后再调用，这里简单示例用 retain_graph=False
        raise RuntimeError("measure_one_step(backward=True) only used inside train loop")
    sync()
    return time.time() - start

def bench(module, data, target, iters=50, amp_mode="fp32", lr=0.1, loss_fn=None):
    """
    返回 (avg_step_time, last_loss)
    amp_mode: 'fp32' | 'fp16' | 'bf16'（不支持时自动回退 fp32）
    """
    device = next(module.parameters()).device
    if loss_fn is None: loss_fn = nn.CrossEntropyLoss().to(device)
    opt = optim.SGD(module.parameters(), lr=lr, momentum=0.9)

    # AMP 配置
    use_amp = amp_mode in ("fp16", "bf16")
    autocast_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
    # 回退：bf16 不支持就当 fp32 处理
    if amp_mode == "bf16" and not torch.cuda.is_bf16_supported():
        use_amp = False

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_mode == "fp16")

    # 预热
    module.train()
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=autocast_dtype if use_amp else None):
            out = module(data)
            loss = loss_fn(out, target)
        if use_amp and amp_mode == "fp16":
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
    sync()

    # 正式计时
    t_acc = 0.0
    last_loss = None
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=autocast_dtype if use_amp else None):
            out = module(data)
            loss = loss_fn(out, target)
        if use_amp and amp_mode == "fp16":
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        sync()
        t_acc += (time.time() - t0)
        last_loss = float(loss.item())
    return t_acc / iters, last_loss

def mk_conv_model(Cin=64, Cout=128, H=128, W=128):
    # 一个中等大小的卷积网络，能触发 cuDNN 多种算法选择
    return nn.Sequential(
        nn.Conv2d(Cin, 128, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, Cout, 3, padding=1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(Cout, 1000)
    )

def mk_lstm_model(input_size=1024, hidden=1024, layers=2):
    # cuDNN LSTM（满足条件时会走 cuDNN 实现）
    return nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=layers, batch_first=True)

def run_suite(args):
    assert torch.cuda.is_available(), "CUDA 不可用"
    device = torch.device("cuda:0")

    env_info()

    # 数据样本（Conv2d）
    N, C, H, W = args.batch, args.cin, args.res, args.res
    x = torch.randn(N, C, H, W, device=device)
    y = torch.randint(0, args.num_classes, (N,), device=device)

    # 数据样本（LSTM）
    B, T, F = args.batch_lstm, args.t_lstm, args.f_lstm
    x_l = torch.randn(B, T, F, device=device)
    y_l = torch.randn(B, T, args.hidden_lstm, device=device)  # 回归型示例

    # =============== Conv2d 基准 ===============
    print("=== Conv2d Forward+Backward ===")
    for use_cudnn in (True, False):
        for use_bench in (True, False):
            torch.backends.cudnn.enabled = use_cudnn
            torch.backends.cudnn.benchmark = use_bench
            for amp in ["fp32", "fp16", "bf16"]:
                if amp == "bf16" and not torch.cuda.is_bf16_supported():
                    continue
                model = mk_conv_model(Cin=args.cin, Cout=args.cout, H=args.res, W=args.res).to(device)
                t, loss = bench(model, x, y, iters=args.iters, amp_mode=amp, lr=args.lr)
                print(f"[Conv2d] cudnn={{enabled:{use_cudnn}, bench:{use_bench}}} amp={amp:<4} "
                      f"=> {t*1000:.2f} ms/step  (loss {loss:.4f})")
    print()

    # =============== LSTM 基准 ===============
    print("=== LSTM Forward+Backward ===")
    loss_mse = nn.MSELoss().to(device)
    for use_cudnn in (True, False):
        for use_bench in (True, False):
            torch.backends.cudnn.enabled = use_cudnn
            torch.backends.cudnn.benchmark = use_bench
            for amp in ["fp32", "fp16", "bf16"]:
                if amp == "bf16" and not torch.cuda.is_bf16_supported():
                    continue
                model = mk_lstm_model(input_size=F, hidden=args.hidden_lstm, layers=args.layers_lstm).to(device)
                # 给 LSTM 后接一个线性，形成监督
                head = nn.Linear(args.hidden_lstm, args.hidden_lstm).to(device)
                seq = nn.Sequential(model, nn.Flatten(0,1), head)  # 简单拼装以触发 backward
                # 构造“标签”
                target = y_l.view(-1, args.hidden_lstm)

                # 自定义 bench：因为 LSTM 返回 (out, (hn, cn))
                use_amp = amp in ("fp16", "bf16")
                autocast_dtype = torch.float16 if amp == "fp16" else torch.bfloat16
                if amp == "bf16" and not torch.cuda.is_bf16_supported():
                    use_amp = False
                opt = optim.SGD(seq.parameters(), lr=args.lr, momentum=0.9)
                scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp == "fp16"))

                # 预热
                for _ in range(5):
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda', enabled=use_amp, dtype=autocast_dtype if use_amp else None):
                        out, _ = model(x_l)
                        pred = head(out.reshape(-1, out.size(-1)))
                        loss = loss_mse(pred, target)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    else:
                        loss.backward(); opt.step()
                torch.cuda.synchronize()

                # 计时
                t_acc = 0.0; last_loss = None
                for _ in range(args.iters_lstm):
                    opt.zero_grad(set_to_none=True)
                    t0 = time.time()
                    with torch.amp.autocast('cuda', enabled=use_amp, dtype=autocast_dtype if use_amp else None):
                        out, _ = model(x_l)
                        pred = head(out.reshape(-1, out.size(-1)))
                        loss = loss_mse(pred, target)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    else:
                        loss.backward(); opt.step()
                    torch.cuda.synchronize()
                    t_acc += (time.time() - t0)
                    last_loss = float(loss.item())
                print(f"[LSTM ] cudnn={{enabled:{use_cudnn}, bench:{use_bench}}} amp={amp:<4} "
                      f"=> {t_acc/args.iters_lstm*1000:.2f} ms/step  (loss {last_loss:.4f})")
    print()

    print("结论参考：")
    print("- 在相同设置下，若 cudnn.enabled=True 明显更快，说明 cuDNN 正在发挥作用；")
    print("- 对 Conv2d，cudnn.benchmark=True 通常会更快（会做算法选择）；")
    print("- AMP(fp16/bf16) 在 Ada/Blackwell 等新卡上通常能进一步提速；")
    print("- LSTM 在满足条件时会走 cuDNN RNN 实现，若 True 比 False 快，说明 cuDNN RNN 正常。")

def parse_args():
    p = argparse.ArgumentParser("cuDNN 体检脚本（Conv2d / LSTM 微基准）")
    # Conv2d 配置
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--cin", type=int, default=64)
    p.add_argument("--cout", type=int, default=128)
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--iters", type=int, default=50)
    # LSTM 配置
    p.add_argument("--batch-lstm", type=int, default=64)
    p.add_argument("--t-lstm", type=int, default=128)     # 序列长度
    p.add_argument("--f-lstm", type=int, default=1024)    # 特征维
    p.add_argument("--hidden-lstm", type=int, default=1024)
    p.add_argument("--layers-lstm", type=int, default=2)
    p.add_argument("--iters-lstm", type=int, default=30)
    # 通用
    p.add_argument("--lr", type=float, default=0.1)
    return p.parse_args()

if __name__ == "__main__":
    try:
        if not torch.cuda.is_available():
            print("❌ CUDA 不可用：请在有 GPU 的环境下运行（WSL2 + NVIDIA 驱动 + 正确的 PyTorch CUDA 构建）。")
        else:
            run_suite(parse_args())
    except Exception as e:
        print("[ERROR]", repr(e))
        traceback.print_exc()


