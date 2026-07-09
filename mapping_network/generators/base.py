from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ParameterGenerator(nn.Module, ABC):
    """参数生成网络基类。负责生成 theta_hat 以及相关的辅助量。

    子类应通过 nn.Linear、nn.Conv2d 等标准模块定义生成网络结构。
    固定参数通过 requires_grad=False 或 register_buffer 管理。
    子类可以重写 light_state_dict() 和 load_light_state_dict() 来控制 checkpoint 的保存/恢复。
    """

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """返回 theta_hat [P']，P' 是目标网络压缩后的总参数数。"""
        pass

    @abstractmethod
    def noisy_forward(self, sigma: float) -> torch.Tensor:
        """对隐变量加高斯噪声后前向，返回 theta_noisy（用于 L_stab）。"""
        pass

    @abstractmethod
    def smooth_loss(self) -> torch.Tensor:
        """返回 L_smooth = ||nabla_z M(z)||^2_F / (P * d)。"""
        pass

    @abstractmethod
    def align_loss(self) -> torch.Tensor:
        """返回 L_align = 1 - cos(z, mean(W_mod, dim=0))。"""
        pass

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fixed_params_count(self) -> int:
        """返回固定（不可训练）参数的总数，用于日志输出。

        默认统计所有 requires_grad=False 的 buffer 和 parameter。
        子类可以重写以提供更精确的统计。
        """
        count = 0
        for p in self.parameters():
            if not p.requires_grad:
                count += p.numel()
        for buf in self.buffers():
            count += buf.numel()
        return count

    # ===== Checkpoint 恢复接口 =====
    # 大 buffer（如 W_fixed, W_mod）不需要存入 checkpoint，可由 w_seed 重建。
    # 子类通过以下方法控制哪些 buffer 需要排除、如何重建。

    def _rebuild_buffers(self):
        """从 w_seed 等属性重建大 buffer（在 load_light_state_dict 前调用）。

        子类如果用了 w_seed 初始化大 buffer，应重写此方法。
        默认不做任何事（适用于没有大 buffer 的生成器）。
        """
        pass

    def light_state_dict(self) -> dict:
        """返回不含大 buffer 的 state_dict，用于 checkpoint 保存。

        默认返回完整 state_dict。子类应重写以排除大 buffer。
        """
        return self.state_dict()

    def load_light_state_dict(self, state_dict: dict):
        """加载 light_state_dict（不含大 buffer），自动处理缺失 key。

        先调用 _rebuild_buffers() 重建大 buffer，然后 strict=False 加载。
        子类可以重写以实现更精确的加载逻辑。
        """
        self._rebuild_buffers()
        self.load_state_dict(state_dict, strict=False)
