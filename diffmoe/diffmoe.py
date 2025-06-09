import math

import einx
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class EMAParameter(nn.Module):
    def __init__(self, size, beta=0.95):
        super().__init__()
        self.parameter = nn.Parameter(torch.empty(size), requires_grad=False)
        self.is_initted = nn.Parameter(
            torch.zeros(1, dtype=torch.bool), requires_grad=False
        )
        self.beta = beta

    def forward(self, x):
        training = self.training
        needs_init = self.is_initted == 0

        if training and needs_init:
            self.parameter.copy_(x.to(self.parameter.dtype))
            self.is_initted.fill_(1)

        elif training:
            p = self.parameter.float()
            x = x.float()
            # Lerp in float32
            self.parameter.copy_(p.lerp(x, 1 - self.beta).to(self.parameter.dtype))

        return self.parameter


class DiffMoeMLP(nn.Module):
    """
    DiffMOE as in:
    DiffMoE: Dynamic Token Selection for Scalable Diffusion Transformers
    https://arxiv.org/pdf/2503.14487
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_experts: int = 8,
        mlp_ratio: int = 4,
        use_mlp_bias: bool = True,
        training_capacity: float = 1.0,
        norm_module: nn.Module | None = None,
        activation_fn: nn.Module | None = None,
    ):
        super().__init__()

        if norm_module is None:
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = norm_module

        self.gate_proj = nn.Linear(embed_dim, num_experts, bias=False)

        self.capacity_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(embed_dim, num_experts),
        )
        self.capacity_predictor_thresholds = EMAParameter(num_experts)

        self.fc1s = nn.Parameter(
            torch.empty(
                num_experts,
                embed_dim * mlp_ratio,
                embed_dim,
            )
        )
        init.kaiming_uniform_(self.fc1s, a=math.sqrt(5))

        if activation_fn is None:
            self.activation_fn = nn.GELU(approximate="tanh")
        else:
            self.activation_fn = activation_fn

        self.fc2s = nn.Parameter(
            torch.empty(
                num_experts,
                embed_dim,
                embed_dim * mlp_ratio,
            )
        )
        init.kaiming_uniform_(self.fc2s, a=math.sqrt(5))

        self.b1s = self.b2s = None
        if use_mlp_bias:
            self.b1s = nn.Parameter(torch.empty(num_experts, embed_dim * mlp_ratio))

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fc1s)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.b1s, -bound, bound)

            self.b2s = nn.Parameter(torch.empty(num_experts, embed_dim))

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fc2s)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.b2s, -bound, bound)

        self.training_capacity = training_capacity

    @property
    def num_experts(self):
        return self.fc1s.shape[0]

    def forward_gate(self, x):
        # TODO
        # this differs from the official diff-moe implementation
        # I derive gate scores from unnormalized x, instead of
        # normalized x
        # And I use tanh scaled to [0,1], instead of softmax
        scores = self.gate_proj(x)
        # scores = scores.softmax(-1)
        scores = (F.tanh(scores) + 1) / 2
        return scores

    def compute_capacity_predictor_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ):
        bs, _ = x.shape
        logits = self.capacity_predictor(x.detach())

        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if padding_mask is not None:
            loss = loss[~padding_mask].mean()
        else:
            loss = loss.mean()

        # k is the total number of MLP forward passes over all experts
        k = int(bs * self.training_capacity) // self.num_experts

        # Find the thresholds such that there are k logits > thresholds
        # bs n -> n
        thresholds = torch.quantile(logits.detach(), 1 - k / bs, dim=0)

        if dist.is_initialized():
            dist.all_reduce(thresholds, op=dist.ReduceOp.SUM)
            thresholds = thresholds / dist.get_world_size()

        # EMA update
        self.capacity_predictor_thresholds(thresholds)

        return loss

    def forward_training(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ):
        """
        x: Shape: (*leading d)
        padding_mask: Optional, shape: (*leading)

        Where padding_mask has the same *leading dimensions as x,
        and contains True where tokens are padding and should
        not be used in the MLP

        Notation:
        k: the total number of selected tokens
        m: the total number of dropped tokens
        d: input hidden channel size
        dd: mlp hidden channel size
        """

        og_shape = x.shape
        device, dtype = x.device, x.dtype

        x = einx.rearrange("... d -> (...) d", x)
        bs = x.shape[0]

        scores = self.forward_gate(x)

        if padding_mask is not None:
            padding_mask = einx.rearrange("... -> (...)", padding_mask)
            assert bs == padding_mask.shape[0]
            mask_value = -200
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), mask_value)

        # k is the total number of MLP forward passes over all experts
        k = int(bs * self.training_capacity) // self.num_experts

        sorted_expert_weights, sorted_expert_idx = scores.sort(0, descending=True)
        kept_expert_weights, dropped_expert_weights = (
            sorted_expert_weights[:k],
            sorted_expert_weights[k:],
        )
        kept_expert_idx, dropped_expert_idx = (
            sorted_expert_idx[:k],
            sorted_expert_idx[k:],
        )

        # Compute capacity predictor loss
        keep_mask = torch.zeros(bs, self.num_experts, device=device, dtype=dtype)

        ones = torch.ones(k, self.num_experts, device=device, dtype=dtype)
        # for i in range (k)
        # for j in range (n)
        # keep_mask[kept_expert_idx[i,j], j] = ones[i,j]
        keep_mask.scatter_(0, kept_expert_idx, ones)

        capacity_predictor_loss = self.compute_capacity_predictor_loss(
            x, keep_mask, padding_mask
        )

        kept_expert_idx = einx.rearrange("k n -> (k n)", kept_expert_idx)

        # [b] d, (k n) -> (k n) d
        y = torch.index_select(x, 0, kept_expert_idx)

        y = self.norm(y)

        y = einx.dot("(k n) d, n dd d -> (k n) dd", y, self.fc1s)
        if self.b1s is not None:
            y = einx.add("(k n) dd, n dd -> (k n) dd", y, self.b1s)

        y = self.activation_fn(y)

        y = einx.dot("(k n) dd, n d dd -> (k n) d", y, self.fc2s)
        if self.b2s is not None:
            y = einx.add("(k n) d, n d -> (k n) d", y, self.b2s)

        y = einx.multiply("(k n) d, k n -> (k n) d", y, kept_expert_weights)
        y = y.to(x.dtype)

        x = torch.index_add(x, 0, kept_expert_idx, y)

        x = x.reshape(og_shape)

        return (x, capacity_predictor_loss)

    def forward_eval_old(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ):
        og_shape = x.shape
        device, dtype = x.device, x.dtype

        x = einx.rearrange("... d -> (...) d", x)
        bs = x.shape[0]

        capacity_logits = self.capacity_predictor(x.detach())
        capacity_thresholds = self.capacity_predictor_thresholds(capacity_logits)
        # bs n
        keep_mask = capacity_logits > capacity_thresholds

        if padding_mask is not None:
            keep_mask.masked_fill_(padding_mask.unsqueeze(-1), False)

        resids = torch.zeros_like(x)

        # TODO this does not work well with torch.compile
        for i in range(self.num_experts):
            # bs n -> bs
            expert_mask = keep_mask[:, i]
            # bs d, bs -> m d
            x_selected = x[expert_mask]

            # m d -> m n
            scores = self.forward_gate(x_selected)
            # m n -> m
            scores = scores[:, i]

            # Forward this expert's mlp
            x_selected = self.norm(x_selected)
            fc1 = self.fc1s[i]
            x_selected = einx.dot("m d, dd d -> m dd", x_selected, fc1)

            if self.b1s is not None:
                b1 = self.b1s[i]
                x_selected = einx.add("m dd, dd", x_selected, b1)

            x_selected = self.activation_fn(x_selected)

            fc2 = self.fc2s[i]
            x_selected = einx.dot("m dd, d dd -> m d", x_selected, fc2)

            if self.b2s is not None:
                b2 = self.b2s[i]
                x_selected = einx.add("m d, d", x_selected, b2)

            x_selected = einx.multiply("m d, m", x_selected, scores)

            resids[expert_mask] += x_selected

        x = x + resids
        x = x.reshape(og_shape)
        return x

    def forward_eval(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        og_shape = x.shape
        device, dtype = x.device, x.dtype

        x_flat = einx.rearrange("... d -> (...) d", x)
        bs = x_flat.shape[0]

        capacity_logits = self.capacity_predictor(x_flat.detach())
        capacity_thresholds = self.capacity_predictor_thresholds(capacity_logits)
        keep_mask = capacity_logits > capacity_thresholds

        if padding_mask is not None:
            padding_mask_flat = einx.rearrange("... -> (...)", padding_mask)
            keep_mask = keep_mask & (~padding_mask_flat).unsqueeze(-1)

        # Precompute gate scores for all tokens
        gate_scores = self.forward_gate(x_flat)

        # If no tokens are activated, return early
        if not keep_mask.any():
            return x.reshape(og_shape)

        # Prepare padded tensors (experts x max_capacity)
        max_capacity = bs  # Safe upper bound
        padded_indices = torch.full(
            (self.num_experts, max_capacity), -1, dtype=torch.long, device=device
        )
        padded_scores = torch.zeros(
            (self.num_experts, max_capacity), dtype=gate_scores.dtype, device=device
        )

        # Gather indices and scores for each expert
        for expert_idx in range(self.num_experts):
            expert_mask = keep_mask[:, expert_idx]
            indices = torch.where(expert_mask)[0]
            num_activated = len(indices)

            if num_activated > 0:
                # Pad indices and scores to max_capacity
                padded_indices[expert_idx, :num_activated] = indices
                padded_scores[expert_idx, :num_activated] = gate_scores[
                    indices, expert_idx
                ]

        # Gather token values with padding protection
        x_padded = torch.cat(
            [x_flat, torch.zeros(1, x_flat.shape[-1], device=device, dtype=dtype)]
        )
        tokens = x_padded[padded_indices]  # [num_experts, max_capacity, d]

        # Process all experts in parallel
        tokens = self.norm(tokens)
        tokens = einx.dot("n m d, n dd d -> n m dd", tokens, self.fc1s)

        if self.b1s is not None:
            tokens = einx.add("n m dd, n dd -> n m dd", tokens, self.b1s)

        tokens = self.activation_fn(tokens)
        tokens = einx.dot("n m dd, n d dd -> n m d", tokens, self.fc2s)

        if self.b2s is not None:
            tokens = einx.add("n m d, n d -> n m d", tokens, self.b2s)

        # Apply expert-specific weights
        tokens = tokens * padded_scores.unsqueeze(-1)

        # Scatter results back to original positions
        flat_output = einx.rearrange("n m d -> (n m) d", tokens)
        flat_indices = einx.rearrange("n m -> (n m)", padded_indices)
        valid_mask = flat_indices != -1

        if valid_mask.any():
            resids = torch.zeros_like(x_flat)
            resids.index_add_(0, flat_indices[valid_mask], flat_output[valid_mask])
            x_flat = x_flat + resids

        return x_flat.reshape(og_shape)

    def forward(self, x, padding_mask=None):
        if self.training:
            return self.forward_training(x=x, padding_mask=padding_mask)
        else:
            return self.forward_eval(x=x, padding_mask=padding_mask)
