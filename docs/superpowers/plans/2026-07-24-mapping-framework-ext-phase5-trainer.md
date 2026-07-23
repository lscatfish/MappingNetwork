# Phase 5: mapping.trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现通用训练器，支持 SLVT/LWT 双模式，Python 配置驱动。

**Architecture:** 新框架中网络自身管理参数生成（Sequential 或逐层 generator），trainer 只需：收集 generators → 前向 → MappingLoss → 反向 → 优化。BaseTrainer 封装训练循环/checkpoint/日志；SLVTTrainer 和 LWTTrainer 分别处理单 generator 和多 generator 场景。

**Tech Stack:** PyTorch, tqdm, pytest, uv

## Global Constraints

- Python 配置（dict/dataclass），不用 yaml
- 只依赖新 `mapping` 包的公共 API
- 测试使用 `device` fixture，禁止硬编码 device
- 禁止修改 `mapping_network/` 旧包
- 测试中用极小网络 + 1-2 个 batch 验证训练循环正确性

## File Structure

| 文件 | 职责 |
|------|------|
| `mapping/trainer/__init__.py` | 导出 BaseTrainer, SLVTTrainer, LWTTrainer |
| `mapping/trainer/base.py` | BaseTrainer：训练循环、评估、checkpoint、日志 |
| `mapping/trainer/slvt.py` | SLVTTrainer：Sequential 网络的训练 |
| `mapping/trainer/lwt.py` | LWTTrainer：逐层 generator 网络的训练 |
| `mapping/trainer/optim_utils.py` | optimizer/scheduler 工厂 |
| `tests/test_trainer.py` | 训练器测试 |

---

### Task 1: optim_utils + BaseTrainer

**设计:**
- `optim_utils.py`: 沿用旧实现的 `build_optimizer` / `build_scheduler`，接口不变
- `BaseTrainer`: 接收 `net`（nn.Module）、`loss_fn`（MappingLoss）、`generators`（列表）、train/test loader
  - `train()`: epoch 循环，每 epoch 调 `train_epoch()` + `evaluate()`
  - `train_epoch()`: 遍历 loader，前向 → loss → 反向 → 梯度裁剪 → step
  - `evaluate()`: no_grad 前向，返回 accuracy
  - `save_checkpoint()` / `load_checkpoint()`: 保存 net + optimizer + scheduler + loss_fn state
  - 可训练参数 = 所有 generator 参数 + loss_fn 的 lambda 参数

### Task 2: SLVTTrainer + LWTTrainer

**设计:**
- `SLVTTrainer(BaseTrainer)`: net 为 Sequential，generators = [net.generator]
- `LWTTrainer(BaseTrainer)`: net 为用户自定义 Module（内含多个 MappingLayer），generators 从 net 中收集所有 MappingLayer.generator
- 两者主要区别在于 generators 的收集方式和 checkpoint 元数据

### Task 3: 测试 + 导出

- 用极小 Sequential 网络（Conv2d(1,4,3) + Flatten + Linear）+ 随机数据验证 SLVT 训练 1 epoch 后 z 被更新
- 用极小 LWT 网络验证逐层训练
- 验证 checkpoint 保存/加载后前向输出一致
- 更新 `mapping/__init__.py` 导出
