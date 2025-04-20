import torch
import torch.nn as nn
import numpy as np
from diffusers import DDPMScheduler
import os
import warnings # For load_state_dict info

# --- Normalize function (Seems Optimal) ---
def normalize(x: torch.Tensor, dim=None, eps=1e-4, dtype=torch.float32) -> torch.Tensor:
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=dtype)
    norm = norm.clamp(min=eps)
    return x / norm.to(x.dtype)

# --- FourierFeatureExtractor (Seems Optimal) ---
class FourierFeatureExtractor(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1, dtype=torch.float32):
        super().__init__()
        self.num_channels = num_channels
        self.bandwidth = bandwidth
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))
        self.dtype=dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 1:
            if x.ndim == 2 and x.shape[1] == 1:
                x = x.squeeze(1)
            else:
                raise ValueError(f"FourierFeatureExtractor expects input shape [batch_size], got {x.shape}")

        y = x.to(self.dtype)
        y = torch.add(torch.mul(y.unsqueeze(-1), self.freqs.to(self.dtype)), self.phases.to(self.dtype))
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

# --- NormalizedLinearLayer (Seems Optimal for EDM2 spec) ---
class NormalizedLinearLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(), dtype=torch.float32):
        super().__init__()
        self.out_channels = out_channels
        k = kernel if isinstance(kernel, tuple) else (kernel,)
        self.kernel_dims = len(k)
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *k))
        self.dtype=dtype

    def forward(self, x: torch.Tensor, gain=1) -> torch.Tensor:
        w = self.weight.to(self.dtype)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w, dtype=self.dtype)) # forced WN
        w = normalize(w, dtype=self.dtype) # standard WN

        fan_in = np.prod(w.shape[1:])
        w = w * (gain / np.sqrt(fan_in)) # magnitude-preserving scaling
        w = w.to(x.dtype)

        if self.kernel_dims == 0: # Linear
             if x.ndim > 2 and x.shape[1] == self.weight.shape[1]:
                 x = x.flatten(1)
             elif x.ndim == 1 and self.weight.shape[1] == 1:
                 x = x.unsqueeze(1)
             elif x.ndim != 2 or x.shape[1] != self.weight.shape[1]:
                  raise ValueError(f"Input shape {x.shape} incompatible with Linear weight shape {self.weight.shape}")
             return x @ w.t()
        elif self.kernel_dims == 2: # Conv2d
            padding_val = w.shape[-1] // 2
            return torch.nn.functional.conv2d(x, w, padding=padding_val)
        else:
             raise NotImplementedError(f"Kernel dimensions {self.kernel_dims} not supported")


# --- AdaptiveLossWeightMLP (Retaining Element-wise Loss) ---
class AdaptiveLossWeightMLP(nn.Module):
    def __init__(
            self,
            noise_scheduler: DDPMScheduler,
            logvar_channels: int = 128,
            lambda_weights: torch.Tensor = None, # Optional precomputed lambda(sigma) weights
            device='cuda',
            dtype=torch.float32,
        ):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.logvar_channels = logvar_channels
        self.dtype = dtype
        self.device = device

        num_timesteps = noise_scheduler.config.num_train_timesteps
        self.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=device, dtype=dtype)
        # Clamp sigma during calculation to avoid issues, but store unclamped for potential use elsewhere
        self.sigmas = ((1.0 - self.alphas_cumprod).sqrt()).to(device=device, dtype=dtype)
        safe_sigmas = self.sigmas.clamp(min=1e-9) # Use clamped for log

        self.register_buffer('precomputed_c_noise', 0.25 * torch.log(safe_sigmas))

        self.logvar_fourier = FourierFeatureExtractor(logvar_channels, dtype=dtype)
        self.logvar_linear = NormalizedLinearLayer(logvar_channels, 1, kernel=(), dtype=dtype)

        # Handle lambda weights
        if lambda_weights is not None:
             if lambda_weights.shape[0] != num_timesteps:
                 raise ValueError(f"Provided lambda_weights shape {lambda_weights.shape} does not match num_timesteps {num_timesteps}")
             self.register_buffer('lambda_weights', lambda_weights.to(device=device, dtype=dtype))
             print("Using provided lambda_weights")
        else:
             self.register_buffer('lambda_weights', torch.ones(num_timesteps, device=device, dtype=dtype))
             print("Defaulting lambda_weights to ones")

    def _forward(self, timesteps: torch.Tensor):
        timesteps = timesteps.long()

        c_noise = self.precomputed_c_noise[timesteps]

        c_noise = c_noise.squeeze()
        if c_noise.ndim == 0:
            c_noise = c_noise.unsqueeze(0)

        fourier_features = self.logvar_fourier(c_noise)
        mlp_output = self.logvar_linear(fourier_features)

        return mlp_output.squeeze() # Shape [batch_size]

    def forward(self, loss: torch.Tensor, timesteps: torch.Tensor):
        """
        Applies adaptive loss weighting while preserving the original loss tensor shape.

        Args:
            loss (torch.Tensor): The input loss tensor (e.g., from UNet). Shape [B, ...].
            timesteps (torch.Tensor): Timesteps for each batch item. Shape [B].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - total_loss (torch.Tensor): Loss scaled and shifted, same shape as input loss.
                - loss_scaled (torch.Tensor): Loss scaled only (for logging), same shape as input loss.
        """
        timesteps = timesteps.long()

        adaptive_loss_weights = self._forward(timesteps)
        loss_scaled = loss * (self.lambda_weights[timesteps] / torch.exp(adaptive_loss_weights)) # type: torch.Tensor
        loss = loss_scaled + adaptive_loss_weights # type: torch.Tensor

        # Return the element-wise losses
        return loss, loss_scaled.detach() # Detach scaled loss for logging

    # --- Rest of methods remain the same ---
    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype=None, metadata=None):
        if metadata is not None and len(metadata) == 0:
            metadata = None
        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            try:
                from safetensors.torch import save_file
                try:
                     from library import train_util
                     if metadata is None: metadata = {}
                     model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
                     metadata["sshs_model_hash"] = model_hash
                     metadata["sshs_legacy_hash"] = legacy_hash
                except ImportError:
                     print("library.train_util not found, saving safetensors without precalculated hashes.")

                save_file(state_dict, file, metadata)
            except ImportError:
                 print("Safetensors not found. Saving as .bin")
                 torch.save(state_dict, os.path.splitext(file)[0] + ".bin")
            except Exception as e:
                 print(f"Error saving safetensor: {e}. Saving as .bin")
                 torch.save(state_dict, os.path.splitext(file)[0] + ".bin")
        else:
            torch.save(state_dict, file)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            try:
                from safetensors.torch import load_file
                weights_sd = load_file(file)
            except ImportError:
                print("Safetensors not found. Trying to load as .bin")
                weights_sd = torch.load(os.path.splitext(file)[0] + ".bin", map_location="cpu")
            except Exception as e:
                 print(f"Error loading safetensor: {e}. Trying to load as .bin")
                 weights_sd = torch.load(os.path.splitext(file)[0] + ".bin", map_location="cpu")
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, strict=False)
        if info.missing_keys:
            warnings.warn(f"Missing keys when loading MLP weights: {info.missing_keys}")
        if info.unexpected_keys:
             warnings.warn(f"Unexpected keys when loading MLP weights: {info.unexpected_keys}")
        return info


# --- create_weight_MLP (Unchanged) ---
def create_weight_MLP(noise_scheduler: DDPMScheduler,
                      logvar_channels: int = 128,
                      lambda_weights: torch.tensor = None,
                      optimizer: torch.optim.Optimizer = torch.optim.AdamW,
                      lr: float = 1e-4,
                      optimizer_args: dict = {'weight_decay': 0, 'betas': (0.9,0.99)},
                      dtype=torch.float32,
                      device='cuda'):
    print(f"Creating weight MLP.")
    lossweightMLP = AdaptiveLossWeightMLP(
        noise_scheduler, logvar_channels, lambda_weights, device, dtype,
    )
    MLP_optim = optimizer(lossweightMLP.get_trainable_params(), lr=lr, **optimizer_args)
    return lossweightMLP, MLP_optim