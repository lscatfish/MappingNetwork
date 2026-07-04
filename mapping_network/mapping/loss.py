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
            mapping_net: MappingNetwork instance
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
        # ||nabla_z M_phi(z)||^2_F / P  (normalised by P for cross-architecture transfer)
        # Use jacfwd (forward-mode AD via vmap) for memory-efficient Jacobian of [P] -> [d].
        from torch.func import jacfwd

        def mapping_fn(z_in):
            return torch.tanh(
                (mapping_net.W_fixed + mapping_net.alpha * z_in.unsqueeze(0)) @ z_in
                + mapping_net.b_fixed
            )

        jacobian = jacfwd(mapping_fn)(z)  # [P, d]
        l_smooth = torch.sum(jacobian ** 2) / jacobian.numel()

        W_mod = mapping_net.W_fixed + mapping_net.alpha * z.unsqueeze(0)

        # === L_align: alignment loss (Equation 30) ===
        W_m = W_mod.mean(dim=0)  # [d]
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
