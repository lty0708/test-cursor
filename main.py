import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple


class DualGPUPerformanceTester:
    def __init__(self):
        self.device_count = torch.cuda.device_count()
        self.device_names = []
        self.results = {}

        # 初始化设备信息
        self._setup_devices()

    def _setup_devices(self):
        """初始化设备信息"""
        print("=" * 60)
        print("双卡性能测试开始")
        print("=" * 60)

        for i in range(self.device_count):
            prop = torch.cuda.get_device_properties(i)
            self.device_names.append(prop.name)
            print(f"GPU {i}: {prop.name}")
            print(f"  计算能力: {prop.major}.{prop.minor}")
            print(f"  显存: {prop.total_memory / 1024 ** 3:.1f} GB")

        if self.device_count < 2:
            raise RuntimeError("检测到GPU数量不足2个，无法进行双卡测试")

    def test_basic_computation(self) -> Dict:
        """测试基础计算性能"""
        print("\n1. 基础计算性能测试")
        print("-" * 40)

        results = {}
        matrix_size = 8192
        iterations = 100

        for device_id in range(self.device_count):
            device = torch.device(f'cuda:{device_id}')

            # 矩阵乘法测试
            a = torch.randn(matrix_size, matrix_size, device=device)
            b = torch.randn(matrix_size, matrix_size, device=device)

            start_time = time.time()
            for _ in range(iterations):
                c = torch.matmul(a, b)
            torch.cuda.synchronize(device)
            end_time = time.time()

            gflops = (2 * matrix_size ** 3 * iterations) / ((end_time - start_time) * 1e9)
            results[device_id] = gflops

            print(f"GPU {device_id} ({self.device_names[device_id]}):")
            print(f"  矩阵大小: {matrix_size}x{matrix_size}")
            print(f"  计算性能: {gflops:.2f} GFLOPS")
            print(f"  耗时: {(end_time - start_time):.2f} 秒")

        self.results['computation'] = results
        return results

    def test_memory_bandwidth(self) -> Dict:
        """测试显存带宽"""
        print("\n2. 显存带宽测试")
        print("-" * 40)

        results = {}
        data_size = 1024 * 1024 * 1024  # 1GB
        iterations = 10

        for device_id in range(self.device_count):
            device = torch.device(f'cuda:{device_id}')

            # 创建测试数据
            data = torch.randn(data_size // 4, device=device)  # float32占4字节

            # 测试拷贝带宽
            start_time = time.time()
            for _ in range(iterations):
                copied_data = data.clone()
            torch.cuda.synchronize(device)
            end_time = time.time()

            bandwidth = (data_size * iterations) / ((end_time - start_time) * 1024 ** 3)
            results[device_id] = bandwidth

            print(f"GPU {device_id} ({self.device_names[device_id]}):")
            print(f"  数据大小: {data_size / 1024 ** 3:.1f} GB")
            print(f"  显存带宽: {bandwidth:.2f} GB/s")

        self.results['memory_bandwidth'] = results
        return results

    def test_p2p_communication(self) -> Dict:
        """测试点对点通信"""
        print("\n3. 点对点通信测试")
        print("-" * 40)

        results = {}
        data_size = 256 * 1024 * 1024  # 256MB
        iterations = 20

        # 检查P2P访问支持
        can_access = torch.cuda.can_device_access_peer(0, 1)
        print(f"GPU 0 → GPU 1 P2P支持: {can_access}")

        if can_access:
            torch.cuda.set_device(0)
            torch.cuda.device(1)

        for direction in [(0, 1), (1, 0)]:
            src_device, dst_device = direction

            data = torch.randn(data_size // 4, device=torch.device(f'cuda:{src_device}'))

            start_time = time.time()
            for _ in range(iterations):
                if src_device != dst_device and can_access:
                    with torch.cuda.device(dst_device):
                        received_data = data.to(f'cuda:{dst_device}')
                else:
                    received_data = data.clone()
            torch.cuda.synchronize()
            end_time = time.time()

            bandwidth = (data_size * iterations) / ((end_time - start_time) * 1024 ** 3)
            results[f'{src_device}→{dst_device}'] = bandwidth

            print(f"GPU {src_device} → GPU {dst_device}:")
            print(f"  通信带宽: {bandwidth:.2f} GB/s")

        self.results['p2p_communication'] = results
        return results

    def test_data_parallel(self) -> Dict:
        """测试DataParallel并行训练"""
        print("\n4. DataParallel并行训练测试")
        print("-" * 40)

        # 创建测试模型
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)
                )

            def forward(self, x):
                return self.net(x)

        # 单卡训练基准
        single_device = torch.device('cuda:0')
        model_single = TestModel().to(single_device)
        optimizer_single = torch.optim.Adam(model_single.parameters())

        batch_size = 64
        inputs_single = torch.randn(batch_size, 1024, device=single_device)
        targets_single = torch.randn(batch_size, 128, device=single_device)

        # 单卡训练时间
        start_time = time.time()
        for _ in range(50):
            optimizer_single.zero_grad()
            outputs = model_single(inputs_single)
            loss = nn.MSELoss()(outputs, targets_single)
            loss.backward()
            optimizer_single.step()
        torch.cuda.synchronize()
        single_time = time.time() - start_time

        # 双卡并行训练
        model_parallel = TestModel()
        model_parallel = nn.DataParallel(model_parallel, device_ids=[0, 1])
        model_parallel = model_parallel.cuda()
        optimizer_parallel = torch.optim.Adam(model_parallel.parameters())

        inputs_parallel = torch.randn(batch_size * 2, 1024).cuda()  # 增大batch size
        targets_parallel = torch.randn(batch_size * 2, 128).cuda()

        start_time = time.time()
        for _ in range(50):
            optimizer_parallel.zero_grad()
            outputs = model_parallel(inputs_parallel)
            loss = nn.MSELoss()(outputs, targets_parallel)
            loss.backward()
            optimizer_parallel.step()
        torch.cuda.synchronize()
        parallel_time = time.time() - start_time

        speedup = single_time / parallel_time
        results = {
            'single_gpu_time': single_time,
            'parallel_time': parallel_time,
            'speedup': speedup
        }

        print(f"单卡训练时间: {single_time:.2f} 秒")
        print(f"双卡训练时间: {parallel_time:.2f} 秒")
        print(f"加速比: {speedup:.2f}x")

        self.results['data_parallel'] = results
        return results

    def test_memory_management(self) -> Dict:
        """测试显存管理"""
        print("\n5. 显存管理测试")
        print("-" * 40)

        results = {}

        for device_id in range(self.device_count):
            device = torch.device(f'cuda:{device_id}')
            torch.cuda.set_device(device)

            # 测试大显存分配
            try:
                # 尝试分配大量显存
                total_memory = torch.cuda.get_device_properties(device).total_memory
                test_size = int(total_memory * 0.8) // 4  # 使用80%显存

                start_time = time.time()
                large_tensor = torch.randn(test_size, device=device)
                allocation_time = time.time() - start_time

                # 清理显存
                del large_tensor
                torch.cuda.empty_cache()

                results[device_id] = {
                    'allocation_success': True,
                    'allocation_time': allocation_time,
                    'tested_memory': test_size * 4 / 1024 ** 3  # GB
                }

                print(f"GPU {device_id} 显存测试:")
                print(f"  分配大小: {test_size * 4 / 1024 ** 3:.1f} GB")
                print(f"  分配时间: {allocation_time:.4f} 秒")
                print(f"  状态: 成功")

            except RuntimeError as e:
                results[device_id] = {
                    'allocation_success': False,
                    'error': str(e)
                }
                print(f"GPU {device_id} 显存测试: 失败 - {e}")

        self.results['memory_management'] = results
        return results

    def run_all_tests(self) -> Dict:
        """运行所有测试"""
        print("开始全面双卡性能测试...")

        tests = [
            self.test_basic_computation,
            self.test_memory_bandwidth,
            self.test_p2p_communication,
            self.test_data_parallel,
            self.test_memory_management
        ]

        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"测试 {test_func.__name__} 失败: {e}")
                continue

        self._generate_report()
        return self.results

    def _generate_report(self):
        """生成测试报告"""
        print("\n" + "=" * 60)
        print("双卡性能测试报告")
        print("=" * 60)

        # 计算综合评分
        if all(key in self.results for key in ['computation', 'memory_bandwidth', 'data_parallel']):
            comp_score = sum(self.results['computation'].values()) / len(self.results['computation'])
            mem_score = sum(self.results['memory_bandwidth'].values()) / len(self.results['memory_bandwidth'])
            parallel_score = self.results['data_parallel']['speedup']

            overall_score = (comp_score / 100 + mem_score + parallel_score) / 3

            print("测试结果汇总:")
            print(f"  计算性能评分: {comp_score:.2f} GFLOPS")
            print(f"  显存带宽评分: {mem_score:.2f} GB/s")
            print(f"  并行效率评分: {parallel_score:.2f}x 加速")
            print(f"  综合性能评分: {overall_score:.2f}/10.0")

            if overall_score > 5.0:
                print("✅ 双卡系统测试通过！性能表现良好")
            else:
                print("⚠️ 双卡系统存在性能问题，建议检查配置")
        else:
            print("部分测试未完成，无法生成完整报告")

        print("=" * 60)


def main():
    """主函数"""
    try:
        # 检查CUDA可用性
        if not torch.cuda.is_available():
            print("错误: CUDA不可用")
            return

        # 创建测试器并运行测试
        tester = DualGPUPerformanceTester()
        results = tester.run_all_tests()

        print("\n测试完成！所有测试项已执行完毕")

    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
