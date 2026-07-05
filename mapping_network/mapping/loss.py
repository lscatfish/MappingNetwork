import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingLoss(nn.Module):
    """
    Mapping Loss: Lmap = Ltask + lambda_st * Lstab + lambda_sm * Lsmooth + lambda_al * Lalign  (Equation 26)

    All losses computed via target_net.functional_forward() so gradients flow back to z.
    L_stab does NOT modify target_net parameters — passes theta_noisy directly for functional forward.
    """

    def __init__(
        self,
        sigma_noise: float = 0.0001,
        lambda_st_init: float = 0.1,
        lambda_sm_init: float = 0.1,
        lambda_al_init: float = 0.1,
    ):
        super().__init__()
        self.sigma_noise = sigma_noise
        self.lambda_st = nn.Parameter(torch.tensor(lambda_st_init))
        self.lambda_sm = nn.Parameter(torch.tensor(lambda_sm_init))
        self.lambda_al = nn.Parameter(torch.tensor(lambda_al_init))

    def forward(self, theta_hat, mapping_net, target_net, x, y):
        """
        Args:
            theta_hat: current theta_hat [P] (with gradient)
            mapping_net: ParameterGenerator instance (e.g., LinearMappingNetwork)
            target_net: target network
            x: input [B, 1, 28, 28]
            y: labels [B]
        Returns:
            total_loss, losses_dict
        """
        # === L_task: cross-entropy (Equation 27) ===
        y_hat = target_net.functional_forward(x, theta_hat)
        l_task = F.cross_entropy(y_hat, y)

        # === L_stab: stability loss (Equation 28) ===
        theta_noisy = mapping_net.noisy_forward(self.sigma_noise)
        y_hat_noisy = target_net.functional_forward(x, theta_noisy)
        l_stab = F.mse_loss(y_hat_noisy, y_hat.detach())

        # === L_smooth & L_align 由 generator 自行实现，避免暴露内部细节 ===
        l_smooth = mapping_net.smooth_loss()
        l_align = mapping_net.align_loss()

        # === Total loss ===
        l_st = torch.sigmoid(self.lambda_st)
        l_sm = torch.sigmoid(self.lambda_sm)
        l_al = torch.sigmoid(self.lambda_al)

        total_loss = l_task + l_st * l_stab + l_sm * l_smooth + l_al * l_align

        losses = {
            'task': l_task.item(),
            'stab': l_stab.item(),
            'smooth': l_smooth.item(),
            'align': l_align.item(),
            'total': total_loss.item(),
        }
        return total_loss, losses
