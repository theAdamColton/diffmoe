import copy
from datetime import datetime
import uuid
from pathlib import Path
import json

import torch
import einx
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import jsonargparse
from dataclasses import dataclass, field
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers.optimization import get_cosine_schedule_with_warmup
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from diffmoe.diffmoe import DiffMoeMLP


@dataclass
class AttentionConfig:
    head_dim: int = 64
    num_heads: int = 2


class Attention(nn.Module):
    def __init__(self, config: AttentionConfig = AttentionConfig()):
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
    hidden_size: int = 128

    attention: AttentionConfig = field(default_factory=lambda: AttentionConfig())


class VanillaDiTBlock(nn.Module):
    def __init__(self, config: VanillaDiTBlockConfig = VanillaDiTBlockConfig()):
        super().__init__()
        self.config = config

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = Attention(config.attention)

        self.norm2 = nn.LayerNorm(config.hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class DiffMoeDiTBlockConfig:
    hidden_size: int = 128
    num_experts: int = 16

    attention: AttentionConfig = field(default_factory=lambda: AttentionConfig())


class DiffMoeDiTBlock(nn.Module):
    def __init__(self, config: DiffMoeDiTBlockConfig = DiffMoeDiTBlockConfig()):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = Attention(config.attention)
        self.mlp = DiffMoeMLP(
            embed_dim=config.hidden_size, num_experts=config.num_experts
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x, *mlp_results = self.mlp(x)
        return x, *mlp_results


@dataclass
class DiTConfig:
    image_channels: int = 1
    image_height: int = 28
    image_width: int = 28

    num_timesteps: int = 1000
    num_cond_embeddings: int = 10
    cond_length: int = 15

    patch_size: int = 4

    num_blocks: int = 4

    use_diff_moe: bool = False
    vanilla_block: VanillaDiTBlockConfig = field(
        default_factory=lambda: VanillaDiTBlockConfig()
    )
    diffmoe_block: DiffMoeDiTBlockConfig = field(
        default_factory=lambda: DiffMoeDiTBlockConfig()
    )


class DiT(nn.Module):
    def __init__(self, config: DiTConfig = DiTConfig()):
        super().__init__()

        self.config = config

        self.hidden_size = config.vanilla_block.hidden_size

        self.input_size = config.image_channels * config.patch_size**2

        self.proj_in = nn.Linear(self.input_size, self.hidden_size)

        self.cond_embedding = nn.Parameter(
            torch.empty(
                config.num_cond_embeddings, config.cond_length, self.hidden_size
            )
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

        if config.use_diff_moe:
            self.blocks = nn.ModuleList(
                DiffMoeDiTBlock(config.diffmoe_block) for _ in range(config.num_blocks)
            )
        else:
            self.blocks = nn.ModuleList(
                VanillaDiTBlock(config.vanilla_block) for _ in range(config.num_blocks)
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
        t = self.timestep_embedding[t]

        c = einx.add("b s d, b d", c, t)

        b, cond_length, d = c.shape
        x = torch.cat((c, x), 1)

        all_capacity_losses = 0
        for block in self.blocks:
            block_result = block(x)

            if config.use_diff_moe:
                x, *mlp_results = block_result
                if self.training:
                    capacity_loss, *_ = mlp_results
                    all_capacity_losses = all_capacity_losses + capacity_loss
            else:
                x = block_result

        all_capacity_losses = all_capacity_losses / config.num_blocks

        c, x = x[:, :cond_length, :], x[:, cond_length:, :]

        x = self.norm_out(x)
        x = self.proj_out(x)
        x = self.unpatchify(x)

        return x, all_capacity_losses


def get_output_path():
    folder_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_str += "_" + str(uuid.uuid4())[:8]
    viz_output_path = Path(".") / "outputs" / folder_str
    viz_output_path.mkdir(parents=True, exist_ok=True)
    return viz_output_path


def load_mnist_data(batch_size):
    """Load MNIST dataset"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader


def generate_samples(
    dit,
    scheduler: DDIMScheduler,
    device,
    autocast_dtype,
    seed=42,
    num_inference_steps=20,
):
    """Generate samples for all digits 0-9"""
    # Generate one sample for each digit
    cond = torch.arange(10, device=device)

    rng = torch.Generator(device).manual_seed(seed)

    b = cond.shape[0]
    noise = torch.randn(
        b,
        dit.config.image_channels,
        dit.config.image_height,
        dit.config.image_width,
        device=device,
        dtype=autocast_dtype,
        generator=rng,
    )

    inference_scheduler = copy.deepcopy(scheduler)
    inference_scheduler.set_timesteps(num_inference_steps)

    for timestep in inference_scheduler.timesteps:
        timestep = timestep.item()
        timestep_tensor = torch.full((b,), timestep, device=device)
        with torch.inference_mode():
            with torch.autocast(device, autocast_dtype):
                noise_pred, *_ = dit(noise, cond, timestep_tensor)

        noise = inference_scheduler.step(
            noise_pred.float(), timestep, noise.float(), generator=rng
        ).prev_sample.to(autocast_dtype)

    return noise


def save_image_grid(images, save_path, epoch):
    """Save a 2x5 grid of generated images"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().float().clip(-1, 1).numpy().squeeze()
        img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Digit {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        save_path / f"generated_epoch_{epoch:03d}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def log_to_jsonl(log_file, **kwargs):
    """Log metrics to JSONL file"""
    kwargs = {
        k: v.item() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
    }
    with open(log_file, "a") as f:
        json.dump(kwargs, f)
        f.write("\n")


def compute_validation_loss(dit, val_loader, scheduler, device, autocast_dtype):
    """Compute validation loss"""
    dit.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            if batch_idx >= 10:  # Only use first 10 batches for speed
                break

            images = images.to(device, dtype=autocast_dtype)
            labels = labels.to(device)

            # Sample random timesteps
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (images.shape[0],),
                device=device,
            )

            # Add noise
            noise = torch.randn_like(images)
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            # Predict noise
            with torch.autocast(device, autocast_dtype):
                noise_pred, *_ = dit(noisy_images, labels, timesteps)

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)
            total_loss += loss.item()
            num_batches += 1

    dit.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def main(
    device: str = "cuda",
    autocast_dtype=torch.bfloat16,
    should_compile: bool = False,
    dit_config: DiTConfig = DiTConfig(),
    validate_every_num_steps: int = 100,
    num_epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 5e-4,
    ema_beta: float = 0.998,
):
    save_path = get_output_path()
    log_file = save_path / "training_log.jsonl"

    print(f"Saving outputs to: {save_path}")

    # Load data
    train_loader, val_loader = load_mnist_data(batch_size)

    dit = DiT(dit_config).to(device)
    ema_dit = DiT(dit_config).to(device)
    ema_dit.load_state_dict(dit.state_dict())
    ema_dit.requires_grad_(False)

    scaler = torch.GradScaler(device)
    optim = torch.optim.AdamW(dit.parameters(), lr=lr, betas=(0.9, 0.95))
    num_train_steps = len(train_loader) * num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=int(num_train_steps * 0.1),
        num_training_steps=int(num_train_steps * 1.25),
    )

    if should_compile:
        dit = torch.compile(dit)
        ema_dit = torch.compile(ema_dit)

    nparameters = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    print("NUM PARAMETERS", nparameters)

    num_denoising_timesteps = dit_config.num_timesteps
    scheduler = DDIMScheduler(
        num_train_timesteps=num_denoising_timesteps,
        rescale_betas_zero_snr=True,
        prediction_type="epsilon",
        beta_schedule="linear",
    )

    global_step = 0

    for epoch in range(num_epochs):
        dit.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, dtype=autocast_dtype)
            labels = labels.to(device)

            # Sample random timesteps
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (images.shape[0],),
                device=device,
            )

            # Add noise
            noise = torch.randn_like(images)
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            # Forward pass
            optim.zero_grad()
            with torch.autocast(device, autocast_dtype):
                noise_pred, capacity_loss, *_ = dit(noisy_images, labels, timesteps)
                diffusion_loss = F.mse_loss(noise_pred, noise)
                loss = diffusion_loss + capacity_loss

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            lr_scheduler.step()

            for p_dst, p_src in zip(ema_dit.parameters(), dit.parameters()):
                if not p_dst.is_floating_point():
                    continue
                p_dst.lerp_(p_src, 1 - ema_beta)

            epoch_loss += diffusion_loss.item()
            num_batches += 1
            global_step += 1

            # Log training step
            log_to_jsonl(
                log_file,
                global_step=global_step,
                epoch=epoch,
                training_loss=loss,
                training_capacity_loss=capacity_loss,
                training_diffusion_loss=diffusion_loss,
                training_lr=lr_scheduler.get_last_lr()[-1],
            )

            # Validation
            if global_step % validate_every_num_steps == 0:
                val_loss = compute_validation_loss(
                    dit, val_loader, scheduler, device, autocast_dtype
                )
                log_to_jsonl(
                    log_file,
                    global_step=global_step,
                    epoch=epoch,
                    validation_loss=val_loss,
                )
                print(f"Step {global_step}, Epoch {epoch}, Val Loss: {val_loss:.4f}")

        avg_epoch_loss = epoch_loss / num_batches

        # Compute validation loss at end of epoch
        val_loss = compute_validation_loss(
            dit, val_loader, scheduler, device, autocast_dtype
        )
        log_to_jsonl(
            log_file,
            global_step=global_step,
            epoch=epoch,
            epoch_training_loss=avg_epoch_loss,
            validation_loss=val_loss,
        )

        print(
            f"Epoch {epoch}, Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Generate samples every 10 epochs
        if epoch % 10 == 0:
            print(f"Generating samples at epoch {epoch}...")
            ema_dit = ema_dit.eval()
            generated_images = generate_samples(
                ema_dit, scheduler, device, autocast_dtype
            )
            save_image_grid(generated_images, save_path, epoch)

    # Generate final samples
    print("Generating final samples...")
    generated_images = generate_samples(dit, scheduler, device, autocast_dtype)
    save_image_grid(generated_images, save_path, num_epochs)

    # Save model
    torch.save(
        {
            "model_state_dict": dit.state_dict(),
            "config": dit_config,
            "epoch": num_epochs,
        },
        save_path / "final_model.pt",
    )

    print(f"Training complete. Outputs saved to: {save_path}")


if __name__ == "__main__":
    jsonargparse.CLI(main)
