#!/usr/bin/env python3
"""
Mamba模型CPU训练验证脚本
验证所有分支在CPU上的推理延迟和兼容性
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

# 添加模型路径
sys.path.append('/root/vitfly/models')
sys.path.append('/root/vitfly/experiments/mamba_branches/branch_B_mambavision_ssm/models')
sys.path.append('/root/vitfly/experiments/mamba_branches/branch_C_cnn_mamba3/models')
sys.path.append('/root/vitfly/experiments/mamba_branches/branch_D_sth_mamba/models')
sys.path.append('/root/vitfly/experiments/mamba_branches/branch_E_decisionmamba/models')

def measure_inference_latency(model, device, batch_size=1, seq_len=1, warmup_iter=10, test_iter=50):
    """测量模型推理延迟"""
    latencies = []
    
    model.eval()
    
    # Warmup
    for _ in range(warmup_iter):
        # 创建测试输入 (使用正确的形状)
        depth_images = torch.randn(batch_size, 60, 90).unsqueeze(1).to(device)  # (B, 1, H, W) - 添加序列维度
        desired_velocities = torch.randn(batch_size, 3).to(device)  # (B, 3) - 3D速度
        quaternions = torch.randn(batch_size, 4).to(device)  # (B, 4)
        X = [depth_images, desired_velocities, quaternions]
        
        with torch.no_grad():
            _ = model(X)
    
    # 实际测试
    for _ in range(test_iter):
        # 重新创建测试输入
        depth_images = torch.randn(batch_size, 60, 90).unsqueeze(1).to(device)  # (B, 1, H, W)  
        desired_velocities = torch.randn(batch_size, 3).to(device)  # (B, 3)
        quaternions = torch.randn(batch_size, 4).to(device)
        X = [depth_images, desired_velocities, quaternions]
    
    # Warmup
    for _ in range(warmup_iter):
        with torch.no_grad():
            _ = model(X)
    
    # 实际测试
    for _ in range(test_iter):
        start_time = time.time()
        with torch.no_grad():
            _ = model(X)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    return np.mean(latencies), np.std(latencies)

def test_cpu_compatibility():
    """测试所有Mamba模型在CPU上的兼容性"""
    print("=== Mamba模型CPU兼容性验证 ===")
    
    test_results = []
    batch_size = 1
    seq_len = 1
    
    # 测试GPU和CPU
    devices = ['cuda', 'cpu']
    
    models_to_test = [
        ('分支B: MambaVision+SSM', 'mambavision_ssm_model', 'MambaVisionSSMNet'),
        ('分支C: CNN+Mamba-3', 'cnn_mamba3_model', 'CNNMamba3Net'),
        ('分支D: STH-Mamba', 'sth_mamba_model', 'STHMambaNet'),
        ('分支E: DecisionMamba', 'decision_mamba_model', 'DecisionMambaNet')
    ]
    
    for model_desc, module, class_name in models_to_test:
        print(f"\n🔍 测试 {model_desc}")
        
        try:
            # 动态导入模块
            module_obj = __import__(module, fromlist=[class_name])
            model_class = getattr(module_obj, class_name)
            
            device_results = []
            
            for device in devices:
                if device == 'cuda' and not torch.cuda.is_available():
                    continue
                
                # 创建模型并移动到设备
                model = model_class()
                model = model.to(device)
                
                # 测量延迟
                latency_mean, latency_std = measure_inference_latency(
                    model, device, batch_size, seq_len
                )
                
                # 计算参数量和内存使用
                param_count = sum(p.numel() for p in model.parameters())
                
                device_results.append({
                    'device': device,
                    'latency_ms': latency_mean,
                    'latency_std': latency_std,
                    'params': param_count
                })
                
                print(f"  {device.upper()}:")
                print(f"    延迟: {latency_mean:.2f}ms (±{latency_std:.2f}ms)")
                print(f"    参数: {param_count:,}")
                print(f"    约束: {'✅' if param_count < 5e6 else '❌'} <5M")
            
            test_results.append({
                'model': model_desc,
                'results': device_results,
                'status': '成功'
            })
            
        except Exception as e:
            print(f"  ❌ 失败: {str(e)[:100]}")
            test_results.append({
                'model': model_desc,
                'results': [],
                'status': f'失败: {str(e)[:100]}'
            })
    
    return test_results

def analyze_performance_results(results):
    """分析性能测试结果"""
    print("\n=== 性能分析报告 ===")
    
    success_count = sum(1 for r in results if r['status'] == '成功')
    print(f"✅ 成功测试模型: {success_count}/{len(results)}")
    
    # 分析延迟性能
    cpu_latencies = []
    gpu_latencies = []
    
    for result in results:
        if result['status'] == '成功':
            for device_result in result['results']:
                if device_result['device'] == 'cpu':
                    cpu_latencies.append(device_result['latency_ms'])
                elif device_result['device'] == 'cuda':
                    gpu_latencies.append(device_result['latency_ms'])
    
    if cpu_latencies:
        avg_cpu_latency = np.mean(cpu_latencies)
        max_cpu_latency = np.max(cpu_latencies)
        print(f"\n📊 CPU性能统计:")
        print(f"   平均延迟: {avg_cpu_latency:.2f}ms")
        print(f"   最大延迟: {max_cpu_latency:.2f}ms")
        print(f"   <5ms目标: {'✅' if avg_cpu_latency < 5 else '❌'}")
    
    if gpu_latencies:
        avg_gpu_latency = np.mean(gpu_latencies)
        print(f"\n📊 GPU性能统计:")
        print(f"   平均延迟: {avg_gpu_latency:.2f}ms")
        print(f"   加速比: {(avg_cpu_latency/avg_gpu_latency):.1f}x")
    
    # 参数量分析
    param_counts = []
    for result in results:
        if result['status'] == '成功' and result['results']:
            param_counts.append(result['results'][0]['params'])
    
    if param_counts:
        avg_params = np.mean(param_counts)
        max_params = np.max(param_counts)
        print(f"\n📊 参数量分析:")
        print(f"   平均参数: {avg_params/1e6:.2f}M")
        print(f"   最大参数: {max_params/1e6:.2f}M")
        print(f"   <5M约束: {'✅' if max_params < 5e6 else '❌'}")

def main():
    print("开始CPU训练验证...")
    
    # 检查设备
    print(f"\n🔧 设备信息:")
    print(f"  CPU核心: {os.cpu_count()}")
    print(f"  GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU设备: {torch.cuda.get_device_name()}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    
    # 运行CPU兼容性测试
    results = test_cpu_compatibility()
    
    # 分析结果
    analyze_performance_results(results)
    
    print("\n=== CPU验证完成 ===")
    
    # 生成验证报告
    success_count = sum(1 for r in results if r['status'] == '成功')
    if success_count == len(results):
        print("✅ 所有模型通过CPU兼容性测试")
    else:
        print(f"⚠️  {success_count}/{len(results)} 个模型通过测试")

if __name__ == "__main__":
    main()