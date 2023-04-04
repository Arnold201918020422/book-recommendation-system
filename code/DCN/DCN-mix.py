import torch
from torch import nn


class CrossNetMix(nn.Module):
    def __init__(self, d_in, n_cross=2, low_rank=32, n_experts=4):
        super().__init__()
        self.d_in = d_in
        self.n_cross = n_cross
        self.low_rank = low_rank
        self.n_experts = n_experts

        # U: (d_in, low_rank)
        self.U_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(n_experts, d_in, low_rank))) for i in range(self.n_cross)])

        # V: (d_in, low_rank)
        self.V_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(n_experts, d_in, low_rank))) for i in range(self.n_cross)])

        # C: (low_rank, low_rank)
        self.C_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
            torch.empty(n_experts, low_rank, low_rank))) for i in range(self.n_cross)])

        # G: (d_in, 1)
        self.gating = nn.ModuleList([nn.Linear(d_in, 1, bias=False) for i in range(self.n_experts)])

        # Bias
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(d_in)) for i in range(self.n_cross)])

    def forward(self, x):
        x0 = x
        xi = x
        for i in range(self.n_cross):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.n_experts):
                # (1) G(xi)
                # compute the gating score by xi
                gating_score_of_experts.append(self.gating[expert_id](xi))

                # (2) E(xi)
                # project the input xi to low_rank space
                v_x = xi @ (self.V_list[i][expert_id])  # (batch_size, low_rank)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = v_x @ self.C_list[i][expert_id]  # (batch_size, low_rank)
                v_x = torch.tanh(v_x)

                # project back to d_in space
                uv_x = v_x @ (self.U_list[i][expert_id].T)  # (batch_size, d_in)
                expert_out = x0 * (uv_x + self.biases[i])
                output_of_experts.append(expert_out)

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (batch_size, d_in, n_experts)
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  # (batch_size, n_experts, 1)
            moe_out = torch.bmm(output_of_experts, gating_score_of_experts.softmax(1))
            xi = torch.squeeze(moe_out) + xi  # (batch_size, d_in)

        return xi
