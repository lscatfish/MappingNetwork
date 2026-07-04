import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingLoss(nn.Module):
    """
    Mapping Loss: Lmap = Ltask + lambda_st * Lstab + lambda_sm * Lsmooth + lambda_al * Lalign  (Equation 26)

    All losses computed via target_net.functional_forward() so gradients flow back to z.
    L_stab does NOT modify target_net parameters — passes theta_noisy directly for functional forward.
    """

    def __init__(self, sigma_noise: float = 0.01):
        super().__init__()
        self.sigma_noise = sigma_noise
        self.lambda_st = nn.Parameter(torch.tensor(0.1))
        self.lambda_sm = nn.Parameter(torch.tensor(0.1))
        self.lambda_al = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, theta_hat, theta_noisy, mapping_net, target_net, x, y):
        """
        Args:
            z: latent vector [d]
            theta_hat: current theta_hat [P] (with gradient)
            theta_noisy: noise-perturbed theta_hat [P] (with gradient, used for L_stab)
            mapping_net: LinearMappingNetwork instance
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
        # Use theta_noisy directly for functional forward, no save/restore of params
        y_hat_noisy = target_net.functional_forward(x, theta_noisy)
        l_stab = F.mse_loss(y_hat_noisy, y_hat.detach())

        # === L_smooth: smoothness loss (Equation 29) ===
        # ||nabla_z M_phi(z)||^2_F / (P * d)
        # M(z) = tanh(W_fixed @ z + alpha * ||z||^2 + b)
        # nabla_z M_i = tanh'(a_i) * (W_fixed[i, :] + 2 * alpha * z)
        # 展开后：||nabla_z M_i||^2 = tanh'(a_i)^2 * (
        #     ||W_fixed[i, :]||^2 + 4*alpha*W_fixed[i, :]@z + 4*alpha^2*||z||^2)
        # 分项计算，避免产生 [P, d] 中间张量导致显存翻倍。
        P, d = mapping_net.P, mapping_net.d
        alpha = mapping_net.alpha
        W_fixed = mapping_net.W_fixed
        b_fixed = mapping_net.b_fixed

        a = W_fixed @ z + alpha * (z * z).sum() + b_fixed  # [P]
        tanh_derivative_sq = (1 - torch.tanh(a) ** 2) ** 2  # [P]

        # term1 = sum_i tanh'(a_i)^2 * ||W_fixed[i, :]||^2
        row_norms_sq = torch.zeros(P, device=z.device, dtype=z.dtype)
        chunk_size = 10000
        for start in range(0, P, chunk_size):
            end = min(start + chunk_size, P)
            row_norms_sq[start:end] = (W_fixed[start:end] ** 2).sum(dim=1)
        term1 = (tanh_derivative_sq * row_norms_sq).sum()

        # term2 = sum_i tanh'(a_i)^2 * 4*alpha*W_fixed[i, :]@z
        #       = 4*alpha * z @ (W_fixed.T @ tanh_derivative_sq)
        term2 = 4 * alpha * (z * (W_fixed.T @ tanh_derivative_sq)).sum()

        # term3 = sum_i tanh'(a_i)^2 * 4*alpha^2*||z||^2
        term3 = 4 * alpha * alpha * (z * z).sum() * tanh_derivative_sq.sum()

        l_smooth = (term1 + term2 + term3) / (P * d)

        # === L_align: alignment loss (Equation 30) ===
        # W_mod.mean(dim=0) = W_fixed.mean(dim=0) + α z
        W_m = mapping_net.W_fixed_mean + mapping_net.alpha * z  # [d]
        cos_sim = F.cosine_similarity(z.unsqueeze(0), W_m.unsqueeze(0))
        l_align = 1 - cos_sim.squeeze()

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
