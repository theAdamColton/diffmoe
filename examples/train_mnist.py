import copy
from datetime import datetime
import uuid
from pathlib import Path

import torch
import einx
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import jsonargparse
from dataclasses import dataclass, field


@dataclass
class AttentionConfig:
    head_dim: int = 64
    num_heads: int = 4


class Attention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

        self.proj_qkv = nn.Linear(
            config.num_heads * config.head_dim,
            3 * config.num_heads * config.head_dim,
            bias=False,
        )

        self.proj_out = nn.Linear(
            config.num_heads * config.head_dim, config.num_heads * config.head_dim
        )

    def forward(self, x):
        qkv = self.proj_qkv(x)
        q, k, v = einx.rearrange(
            "b s (three h d) -> three b h s d", qkv, h=self.config.num_heads, three=3
        )
        x = F.scaled_dot_product_attention(q, k, v)
        x = einx.rearrange("b h s d -> b s (h d)", x)
        x = self.proj_out(x)
        return x


@dataclass
class VanillaDiTBlockConfig:
    hidden_size: int = 256
    cond_size: int = 256

    attention_config: AttentionConfig = field(default_factory=lambda: AttentionConfig())


class VanillaDiTBlock(nn.Module):
    def __init__(self, config: VanillaDiTBlockConfig):
        super().__init__()
        self.config = config

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = Attention(config.attention_config)

        self.norm2 = nn.LayerNorm(config.hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )

    def forward(self, x, cond):
        x = x + self.attn(self.norm1(x, cond))
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class DiTConfig:
    image_channels: int = 3
    image_height: int = 64
    image_width: int = 64

    num_timesteps: int = 1000
    num_cond_embeddings: int = 10

    patch_size: int = 4

    num_blocks: int = 8
    block_config: VanillaDiTBlockConfig = field(
        default_factory=lambda: VanillaDiTBlockConfig()
    )


class DiT(nn.Module):
    def __init__(self, config: DiTConfig = DiTConfig()):
        super().__init__()

        self.config = config

        self.hidden_size = config.block_config.hidden_size

        self.input_size = config.image_channels * config.patch_size**2

        self.proj_in = nn.Linear(self.input_size, self.hidden_size)

        self.cond_embedding = nn.Parameter(
            torch.empty(config.num_cond_embeddings, self.hidden_size)
        )
        init.trunc_normal_(self.cond_embedding, std=0.02)

        self.timestep_embedding = nn.Parameter(
            torch.empty(config.num_timesteps, self.hidden_size)
        )
        init.trunc_normal_(self.timestep_embedding, std=0.02)

        self.height_position_embed = nn.Parameter(
            torch.empty(config.image_height // config.patch_size, self.hidden_size)
        )
        init.trunc_normal_(self.height_position_embed, std=0.02)
        self.width_position_embed = nn.Parameter(
            torch.empty(config.image_width // config.patch_size, self.hidden_size)
        )
        init.trunc_normal_(self.width_position_embed, std=0.02)

        self.blocks = nn.ModuleList(
            VanillaDiTBlock(config.block_config) for _ in range(config.num_blocks)
        )

        self.norm_out = nn.LayerNorm(self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, self.input_size)

    def patchify(self, x):
        return einx.rearrange(
            "... c (nph ph) (npw pw) -> ... (nph npw) (ph pw c)",
            x,
            ph=self.config.patch_size,
            pw=self.config.patch_size,
        )

    def unpatchify(self, x):
        return einx.rearrange(
            "... (nph npw) (ph pw c) -> ... c (nph ph) (npw pw)",
            x,
            ph=self.config.patch_size,
            pw=self.config.patch_size,
            nph=self.config.image_height // self.config.patch_size,
            npw=self.config.image_width // self.config.patch_size,
        )

    def forward(self, x, c, t):
        config = self.config

        x = self.patchify(x)
        x = self.proj_in(x)

        pos_emb = einx.add(
            "h d, w d -> (h w) d", self.height_position_embed, self.width_position_embed
        )
        x = einx.add("... s d, s d", x, pos_emb)

        c = self.cond_embedding[c]
        c = einx.add("... s d, s d -> ... s d", c, self.cond_pos_embedding)

        t = self.timestep_embedding[t]

        cond_length = c.shape[1]
        x = torch.cat((c, x), 1)

        for block in self.blocks:
            x = block(x, t)

        c, x = x[:, :cond_length, :], x[:, cond_length:, :]

        x = self.norm_out(x)
        x = self.proj_out(x)
        x = self.unpatchify(x)

        return x


def get_viz_output_path():
    folder_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder_str += str(uuid.uuid4())
    viz_output_path = Path(".") / "viz-outputs" / folder_str
    viz_output_path.mkdir(parents=True)
    return viz_output_path


def main(
    device: str = "cuda",
    autocast_dtype=torch.bfloat16,
    should_compile: bool = False,
    dit_config: DiTConfig = DiTConfig(),
    validate_every_num_steps: int = 100,
    num_epochs: int = 10,
    batch_size: int = 256,
):
    save_path = get_viz_output_path()

    dit = DiT(dit_config).to(device)

    scaler = torch.GradScaler(device)
    optim = torch.optim.AdamW(dit.parameters(), lr=5e-4, betas=(0.9, 0.95))

    if should_compile:
        dit = torch.compile(dit)

    nparameters = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    print("NUM PARAMETERS", nparameters)

    num_denoising_timesteps = dit_config.num_timesteps
    scheduler = DDIMScheduler(
        num_train_timesteps=num_denoising_timesteps,
        rescale_betas_zero_snr=True,
        prediction_type="epsilon",
    )

    def generate(cond):
        b, *_ = cond.shape
        noise = torch.randn(
            b,
            dit_config.image_channels,
            dit_config.image_height,
            dit_config.image_width,
            device=device,
            dtype=autocast_dtype,
        )

        inference_scheduler = copy.deepcopy(scheduler)
        inference_scheduler.set_timesteps(num_denoising_timesteps)

        for timestep in inference_scheduler.timesteps:
            timestep = timestep.item()
            timestep_tensor = torch.full((b,), timestep, device=device)
            with torch.inference_mode():
                with torch.autocast(device, autocast_dtype):
                    noise_pred = dit(noise, cond, timestep_tensor)

            noise = inference_scheduler.step(
                noise_pred.float(), timestep, noise.float()
            ).prev_sample.to(autocast_dtype)

        return noise


if __name__ == "__main__":
    jsonargparse.CLI(main)
