# NPU优化Qwen2集成指南

本指南展示如何将 `catcoc_matmul_allreduce` 算子集成到 SGLang 的 Qwen2 模型中，实现更高效的分布式推理。

## 概述

`catcoc_matmul_allreduce` 算子将矩阵乘法和 allreduce 操作融合为单个操作，显著减少了通信开销和计算延迟。这个集成专门针对 NPU (Neural Processing Unit) 设备进行了优化。

## 核心特性

- **融合操作**: 将 matmul + allreduce 合并为单个内核调用
- **共享内存优化**: 使用高效的共享内存进行进程间通信
- **NZ格式支持**: 支持压缩权重格式以提高性能
- **自动回退**: 当NPU不可用时自动回退到标准实现
- **零修改集成**: 现有代码无需修改即可受益

## 文件结构

```
sglang/
├── srt/
│   ├── layers/
│   │   └── npu_linear.py          # NPU优化的线性层实现
│   └── models/
│       └── qwen2.py               # 修改后的Qwen2模型
└── test_qwen2_npu.py              # 测试脚本
```

## 安装要求

### 基本依赖
```bash
# SGLang基本安装
pip install sglang

# NPU支持包
pip install torch_npu  # NPU版本的PyTorch
pip install shmem      # 共享内存支持
```

### 编译要求

确保 sgl-kernel 编译时启用了内核模块：
```bash
cd sgl-kernel/
export BUILD_KERNELS_MODULE=1
python setup.py build_ext --inplace
```

## 使用方法

### 1. 自动集成 (推荐)

修改后的 Qwen2 模型会自动检测 NPU 环境并使用优化层：

```python
from sglang.srt.models.qwen2 import Qwen2MLP

# 创建模型 - 会自动选择合适的实现
model = Qwen2MLP(
    hidden_size=2048,
    intermediate_size=5632,
    hidden_act="silu"
)
```

### 2. 手动使用NPU层

如果需要直接使用NPU优化层：

```python
from sglang.srt.layers.npu_linear import CatcocRowParallelLinear

# 创建NPU优化的线性层
layer = CatcocRowParallelLinear(
    input_size=5632,
    output_size=2048,
    bias=False,
    use_nz_format=True,  # 启用NZ压缩格式
    shmem_size=1024 * 1024 * 1024,  # 1GB共享内存
    team_id=0
)
```

## 分布式运行

### 单节点多NPU

```bash
# 2个NPU设备
torchrun --nproc_per_node=2 test_qwen2_npu.py

# 4个NPU设备
torchrun --nproc_per_node=4 your_script.py
```

### 多节点设置

```bash
# 主节点
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=2 \
         --master_addr=192.168.1.100 --master_port=29500 \
         your_script.py

# 从节点
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=2 \
         --master_addr=192.168.1.100 --master_port=29500 \
         your_script.py
```

## 环境配置

### 必需的环境变量

```bash
# 设置共享内存网络接口 (根据实际网络接口调整)
export SHMEM_UID_SOCK_IFNAM="enp194s0f0::inet4"

# NPU配置
export LOCAL_RANK=0
export WORLD_SIZE=2
```

### 可选配置

```bash
# 启用NPU内部格式优化
export NPU_ALLOW_INTERNAL_FORMAT=1

# 禁用JIT编译 (如果遇到编译问题)
export NPU_JIT_COMPILE=0
```

## 性能优化建议

### 1. 共享内存大小

根据模型大小调整共享内存：
```python
# 小模型 (< 7B 参数)
shmem_size = 512 * 1024 * 1024  # 512MB

# 大模型 (7B+ 参数)
shmem_size = 2 * 1024 * 1024 * 1024  # 2GB
```

### 2. NZ格式优化

对于推理场景，启用NZ格式可以显著提升性能：
```python
layer = CatcocRowParallelLinear(
    ...,
    use_nz_format=True,  # 启用权重压缩
)
```

### 3. 批量大小优化

NPU设备通常在较大批量下表现更好：
```python
# 推荐的批量大小范围
batch_size = 8  # 对于较小模型
batch_size = 16 # 对于中等模型
```

## 故障排除

### 常见问题

1. **共享内存初始化失败**
   ```bash
   # 检查网络接口
   ip addr show
   # 更新环境变量
   export SHMEM_UID_SOCK_IFNAM="your_interface::inet4"
   ```

2. **catcoc算子不可用**
   ```bash
   # 确保编译时启用了内核模块
   export BUILD_KERNELS_MODULE=1
   cd sgl-kernel && python setup.py build_ext --inplace
   ```

3. **设备不匹配**
   ```python
   # 确保所有张量都在正确的NPU设备上
   torch.npu.set_device(local_rank)
   tensor = tensor.to(f"npu:{local_rank}")
   ```

### 调试模式

启用详细日志：
```python
import logging
logging.getLogger("sglang.srt.layers.npu_linear").setLevel(logging.DEBUG)
```

## 性能评估

运行基准测试：
```bash
# 标准实现
python test_qwen2_npu.py --baseline

# NPU优化实现
python test_qwen2_npu.py --optimized

# 性能对比
python test_qwen2_npu.py --benchmark
```

## 注意事项

1. **兼容性**: 目前仅支持 FP16 和 BF16 数据类型
2. **量化**: 暂不支持量化模型（会自动回退到标准实现）
3. **单设备**: 单设备情况下会回退到标准实现（无需 allreduce）
4. **内存管理**: 共享内存会在进程结束时自动释放

## 扩展指南

如果需要为其他模型添加类似优化：

1. 识别带有 allreduce 的线性层（通常是 down_proj 层）
2. 使用 `CatcocRowParallelLinear` 替换 `RowParallelLinear`
3. 添加设备检测逻辑
4. 测试性能改进

## 更多信息

- [SGLang 官方文档](https://sgl-lang.readthedocs.io/)
- [NPU 开发指南](https://www.hiascend.com/document)
- [性能优化最佳实践](https://github.com/sgl-project/sglang/docs/performance)
