# Phase 4: mapping.loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现通用 MappingLoss，只依赖 `generator.z` 与 `generator.forward()`，支持 LWT/SLVT 双模式。

**Architecture:** MappingLoss 接收 trunk 网络的 logits 与 target 计算 L_task；对每个 generator 独立计算 L_stab（z 加噪重前向）、L_smooth（Jacobian 范数）、L_align（z 与生成权重均值余弦相似度）。LWT 模式逐层聚合，SLVT 模式整体计算。λ 为可学习参数，sigmoid 门控。

**Tech Stack:** PyTorch (torch.autograd.functional.jacobian), pytest, uv

## Global Constraints

- 只依赖 `generator.z`（nn.Parameter）和 `generator.forward() -> (weight, bias)`，不要求 generator 实现额外方法
- 测试使用 `device` fixture，禁止硬编码 device
- 禁止修改 `mapping_network/` 旧包
- Ruff: line-length=100, 单引号, E/F/I
- 全量测试无回归

## File Structure

| 文件 | 职责 |
|------|------|
| `mapping/loss.py` | MappingLoss 实现 |
| `mapping/__init__.py` | 导出 MappingLoss |
| `tests/test_mapping_loss.py` | 单元测试 + 双模式集成测试 |

---

### Task 1: MappingLoss 核心实现

**Files:**
- Create: `mapping/loss.py`
- Modify: `mapping/__init__.py`
- Test: `tests/test_mapping_loss.py`

**设计:**

```python
class MappingLoss(nn.Module):
    def __init__(self, sigma_noise=1e-4, n_stab_samples=5,
                 lambda_st_init=0.1, lambda_sm_init=0.1, lambda_al_init=0.1):
        # 可学习 λ 参数（sigmoid 门控）

    def forward(self, logits, target, generators):
        """
        Args:
            logits: trunk 网络输出 [B, num_classes]
            target: 标签 [B]
            generators: 单个 Generator 或 Generator 列表
        Returns:
            (total_loss, losses_dict)
        """
```

**各损失计算方式:**
- `L_task`: `F.cross_entropy(logits, target)`
- `L_stab`: 对每个 generator，采样 n 次 `z + σ·ε`，临时替换 z 调用 `generator.forward()` 得到 (w_noisy, b_noisy)，与无噪输出求 MSE（detach 无噪输出）
- `L_smooth`: `‖J‖²_F / (P·d)`，J = ∂(w_flat)/∂z，用 `torch.autograd.functional.jacobian`
- `L_align`: `1 - cos_sim(z, w_flat.mean(dim=0))`（对 w_flat 按列取均值得到 d 维向量）
- 总损失: `L_task + sigmoid(λ_st)·L_stab + sigmoid(λ_sm)·L_smooth + sigmoid(λ_al)·L_align`

**LWT 模式:** generators 为列表，L_stab/L_smooth/L_align 对每个 generator 独立计算后取均值
**SLVT 模式:** generators 为单个 Generator，整体计算

- [ ] **Step 1:** 编写测试（SLVT 单 generator + LWT 多 generator + 梯度回传 + λ 可学习）
- [ ] **Step 2:** 实现 `mapping/loss.py`
- [ ] **Step 3:** 更新 `mapping/__init__.py` 导出
- [ ] **Step 4:** 全量测试 + ruff check
- [ ] **Step 5:** Commit
