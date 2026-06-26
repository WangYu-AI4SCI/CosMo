# save as check_env.py
import torch

print("="*60)
print("环境诊断")
print("="*60)

print(f"当前环境: mocomca")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 检查CUDA编译信息
if hasattr(torch.version, 'cuda'):
    print(f"编译信息: CUDA {torch.version.cuda}")
else:
    print(f"编译信息: CPU版本")

# 如果CUDA可用，显示详细信息
if torch.cuda.is_available():
    print(f"\nGPU详细信息:")
    print(f"  设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    显存: {props.total_memory / 1e9:.2f} GB")
        print(f"    计算能力: {props.major}.{props.minor}")
else:
    print(f"\nCUDA不可用原因:")
    # 尝试显示更多信息
    try:
        config = torch.__config__.show()
        print(f"  构建配置:")
        for line in config.split('\n'):
            if 'CUDA' in line or 'cuda' in line:
                print(f"    {line}")
    except:
        pass

print("="*60)

# 简单测试
print("\n简单测试:")
x = torch.randn(3, 3)
print(f"CPU张量创建成功: {x.shape}")
print(f"CPU张量设备: {x.device}")

if torch.cuda.is_available():
    try:
        x_gpu = x.cuda()
        print(f"GPU张量创建成功: {x_gpu.device}")
        print("✅ GPU环境正常！")
    except Exception as e:
        print(f"❌ GPU测试失败: {e}")
else:
    print("❌ 无法测试GPU，CUDA不可用")

print("="*60)