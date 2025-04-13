import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from torch.distributions import Beta
import os
import matplotlib.pyplot as plt

import logging

import library.train_util as train_util

from library.custom_train_functions import (
    apply_masked_loss,
)

logger = logging.getLogger(__name__)

class TimestepSampler(nn.Module):
    """
    Neural network that parameterizes a Beta distribution for adaptive timestep sampling.
    """
    def __init__(self, in_channels=3, hidden_channels=192, hidden_depth=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Simple encoder to process image features
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(hidden_channels // 4, hidden_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # MLP to output Beta distribution parameters
        layers = []
        layers.append(nn.Linear(hidden_channels, hidden_channels))
        layers.append(nn.SiLU())
        
        for _ in range(hidden_depth - 1):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_channels, 2))  # Output a and b parameters
        layers.append(nn.Softplus())  # Ensure parameters are positive
        
        self.mlp = nn.Sequential(*layers)
        
        logger.info(f"Initialized TimestepSampler with in_channels={in_channels}, hidden_channels={hidden_channels}, hidden_depth={hidden_depth}")
        
    def forward(self, x0):
        """
        Args:
            x0: Original clean image [B, C, H, W]
        
        Returns:
            a, b: Parameters for Beta distribution for timestep sampling
        """
        # Extract features from the image
        features = self.encoder(x0).flatten(1)
        
        # Generate a and b parameters
        params = self.mlp(features) + 1.0  # Add 1.0 to ensure a, b > 0
        a, b = params[:, 0], params[:, 1]
        
        return a, b
    
    @torch.no_grad()
    def sample_timestep(self, x0, min_timestep=0, max_timestep=1000):
        """
        Sample a timestep using the Beta distribution.
        """
        a, b = self.forward(x0)
        
        # Create a Beta distribution and sample from it
        beta_dist = Beta(a, b)
        u = beta_dist.sample()  # Sample in [0, 1]
        
        # Scale to timestep range and convert to integer
        timesteps = (min_timestep + u * ((max_timestep - 1) - min_timestep)).long()
        timesteps = torch.clamp(timesteps, min_timestep, max_timestep)
        
        return timesteps
    
    def log_prob(self, a, b, t, num_timesteps=1000):
        """
        Compute log probability of sampled timesteps.
        """
        beta_dist = Beta(a, b)
        u = t.float() / num_timesteps
        return beta_dist.log_prob(u)
    
    def get_trainable_params(self):
        """Return all trainable parameters for gradient clipping"""
        return self.parameters()
    
    def save_weights(self, file_path, dtype=None, metadata=None):
        """Save the model weights to a file"""
        logger.info(f"Saving TimestepSampler weights to {file_path}")
        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].to(dtype)
                
        if os.path.splitext(file_path)[1] == '.safetensors':
            from safetensors.torch import save_file
            # Convert metadata values to string
            if metadata is not None:
                metadata = {k: str(v) for k, v in metadata.items()}
            save_file(state_dict, file_path, metadata)
        else:
            torch.save(state_dict, file_path)
        
        logger.info(f"Successfully saved TimestepSampler weights")
    
    def load_weights(self, file_path):
        """Load weights from a file"""
        logger.info(f"Loading TimestepSampler weights from {file_path}")
        if os.path.splitext(file_path)[1] == '.safetensors':
            from safetensors.torch import load_file
            state_dict = load_file(file_path)
        else:
            state_dict = torch.load(file_path, map_location='cpu')
            
        self.load_state_dict(state_dict)
        logger.info(f"Successfully loaded TimestepSampler weights")
        return {"state": "Loaded TimestepSampler weights"}

class DeltaApproximator:
    """
    Approximates the impact of gradient updates (Δᵗₖ) using pre-computed 
    predictions before and after network updates.
    """
    def __init__(self, queue_size=20, num_subset=3, num_samples=25):
        self.queue = deque(maxlen=queue_size)
        self.num_subset = num_subset
        self.sampled_timesteps = None
        self.before_predictions = {}  # Store predictions before update
        self.num_samples = num_samples
        logger.info(f"Initialized DeltaApproximator with queue_size={queue_size}, num_subset={num_subset}")
        
    def compute_before_predictions(self, 
                                   args, 
                                   accelerator, 
                                   noise_scheduler, 
                                   latents, 
                                   batch, 
                                   unet, 
                                   text_encoder_conds, 
                                   weight_dtype, 
                                   network_trainer,
                                   min_timestep=0,
                                   max_timestep=1000):
        """
        Compute and store model predictions before the network update.
        Called right before optimizer.step().
        """
        logger.info("Computing 'before' predictions for adaptive timestep sampling")
        batch_size = latents.shape[0]
        device = latents.device
        
        # Reset storage for this update cycle
        self.before_predictions = {}

        # always define min_timestep and max_timestep up-front
        min_timestep = 0 if min_timestep is None else min_timestep
        max_timestep = 1000 if max_timestep is None else max_timestep
        
        # Select timesteps to sample (for efficiency)
        num_samples = min(self.num_samples, max_timestep - min_timestep)  # Sample a reasonable number of timesteps
        self.sampled_timesteps = torch.linspace(min_timestep, max_timestep - 1, num_samples, device="cpu").to(dtype=torch.long, device=device)
        
        # Generate one noise to use for all timesteps
        noise = torch.randn_like(latents)
        
        with torch.no_grad():
            for t_idx, tau in enumerate(self.sampled_timesteps):
                timesteps = torch.full((batch_size,), tau, dtype=torch.long, device=device)

                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, fixed_timesteps=timesteps, train=False)

                # Predict with current network
                noise_pred = network_trainer.call_unet(args, accelerator, unet, noisy_latents, timesteps, text_encoder_conds, batch, weight_dtype)

                if args.v_parameterization:
                    # v-parameterization training
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                if noise_pred.dtype not in {torch.float32, torch.float64}:
                    noise_pred = noise_pred.float()

                if target.dtype not in {torch.float32, torch.float64}:
                    target = target.float()
                
                # Calculate loss
                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                loss = train_util.conditional_loss(noise_pred, target, args.loss_type, "none", huber_c, scale=float(args.loss_scale))

                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean(dim=[1, 2, 3])  # Mean over dimensions

                loss_weights = batch["loss_weights"]  # Sample-wise weights
                loss = loss * loss_weights

                if args.sangoi_loss_modifier:
                    # Min SNR should be zero for zero_terminal_snr
                    if args.zero_terminal_snr:
                        min_snr = 0
                    else:
                        min_snr = float(args.sangoi_loss_modifier_min_snr)

                    loss = loss * train_util.sangoi_loss_modifier(timesteps, 
                                                            noise_pred, 
                                                            target, 
                                                            noise_scheduler,
                                                            min_snr,
                                                            float(args.sangoi_loss_modifier_max_snr))

                # min snr gamma, scale v pred loss like noise pred, v pred like loss, debiased estimation etc.
                loss = network_trainer.post_process_loss(loss, args, timesteps, noise_scheduler)

                if args.loss_multipler or args.loss_multiplier:
                    loss.mul_(float(args.loss_multipler or args.loss_multiplier) if args.loss_multipler is not None or args.loss_multiplier is not None else 1.0)

                loss = loss.mean()  # Mean over batch
                
                # Store for this timestep
                self.before_predictions[tau.item()] = {
                    'loss': loss.detach().item()
                }
            
            logger.info(f"Stored 'before' predictions for {len(self.sampled_timesteps)} timesteps")
    
    def compute_delta_t_k(self, 
                        args, 
                        accelerator, 
                        noise_scheduler, 
                        latents, 
                        batch, 
                        unet, 
                        text_encoder_conds, 
                        weight_dtype,
                        network_trainer):
        """
        Compute δᵗₖ,τ by comparing stored 'before' predictions with new 'after' predictions.
        Called right after optimizer.step().
        """
        logger.info("Computing 'after' predictions and calculating deltas")
        batch_size = latents.shape[0]
        device = latents.device
        dtype = latents.dtype
        delta_t_k = torch.zeros(self.num_timesteps, device=device)
        
        with torch.no_grad():
            for t_idx, tau in enumerate(self.sampled_timesteps):
                timesteps = torch.full((batch_size,), tau, dtype=torch.long, device=device)
                
                # Get stored values
                stored = self.before_predictions[tau.item()]
                loss_before = stored['loss']
                
                # Sample noise, sample a random timestep for each image, and add noise to the latents,
                # with noise offset and/or multires noise if specified
                noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents, fixed_timesteps=timesteps, train=False)

                # Predict with current network
                noise_pred = network_trainer.call_unet(args, accelerator, unet, noisy_latents, timesteps, text_encoder_conds, batch, weight_dtype)

                if args.v_parameterization:
                    # v-parameterization training
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                if noise_pred.dtype not in {torch.float32, torch.float64}:
                    noise_pred = noise_pred.float()

                if target.dtype not in {torch.float32, torch.float64}:
                    target = target.float()
                
                # Calculate loss
                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                loss = train_util.conditional_loss(noise_pred, target, args.loss_type, "none", huber_c, scale=float(args.loss_scale))
                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean(dim=[1, 2, 3])  # Mean over dimensions

                loss_weights = batch["loss_weights"]  # Sample-wise weights
                loss = loss * loss_weights

                if args.sangoi_loss_modifier:
                    # Min SNR should be zero for zero_terminal_snr
                    if args.zero_terminal_snr:
                        min_snr = 0
                    else:
                        min_snr = float(args.sangoi_loss_modifier_min_snr)

                    loss = loss * train_util.sangoi_loss_modifier(timesteps, 
                                                            noise_pred, 
                                                            target, 
                                                            noise_scheduler,
                                                            min_snr,
                                                            float(args.sangoi_loss_modifier_max_snr))

                # min snr gamma, scale v pred loss like noise pred, v pred like loss, debiased estimation etc.
                loss = network_trainer.post_process_loss(loss, args, timesteps, noise_scheduler)

                if args.loss_multipler or args.loss_multiplier:
                    loss.mul_(float(args.loss_multipler or args.loss_multiplier) if args.loss_multipler is not None or args.loss_multiplier is not None else 1.0)

                loss_after = loss.mean()  # Mean over batch
                
                # Compute delta for this timestep
                delta_t_k[tau] = loss_before - loss_after

            # Reset storage
            self.before_predictions = {}
            
            # Interpolate values for non-sampled timesteps
            if len(self.sampled_timesteps) < self.num_timesteps:
                # Get indices and values of sampled timesteps
                indices = self.sampled_timesteps.to(dtype=torch.float32,device="cpu").numpy()
                values = delta_t_k[self.sampled_timesteps].to(dtype=torch.float32,device="cpu").numpy()
                
                # Interpolate for all timesteps
                all_indices = np.arange(self.num_timesteps)
                interp_values = np.interp(all_indices, indices, values)
                
                # Update delta_t_k with interpolated values
                delta_t_k = torch.tensor(interp_values, device=device, dtype=dtype)
            
            # Log the range of delta values
            min_delta = delta_t_k.min().item()
            max_delta = delta_t_k.max().item()
            mean_delta = delta_t_k.mean().item()
            logger.info(f"Delta range: min={min_delta:.6f}, max={max_delta:.6f}, mean={mean_delta:.6f}")
        
        return delta_t_k
    
    def update_queue(self, delta_t_k):
        """Add computed delta_t_k to the queue"""
        if isinstance(delta_t_k, torch.Tensor):
            delta_t_k = delta_t_k.detach().to(dtype=torch.float32,device="cpu").numpy()
        self.queue.append(delta_t_k)
        logger.info(f"Updated delta queue, current size: {len(self.queue)}/{self.queue.maxlen}")
    
    def select_subset(self, min_timestep=0, max_timestep=1000):
        """Feature selection to identify the most important timesteps."""
        logger.info("Selecting subset of most important timesteps")
        if len(self.queue) <= 1:
            # always define min_timestep and max_timestep up-front
            min_timestep = 0 if min_timestep is None else min_timestep
            max_timestep = 1000 if max_timestep is None else max_timestep
            
            # If queue is too small, return evenly spaced timesteps
            subset = np.linspace(min_timestep, max_timestep - 1, self.num_subset, dtype=int)
            logger.info(f"Queue too small, using evenly spaced timesteps: {subset}")
            return subset
        
        # Stack all delta_t_k values from the queue
        delta_matrix = np.stack(list(self.queue), axis=0)  # [queue_size, num_timesteps]
        
        # Use variance-based selection (higher variance = more impactful timesteps)
        variances = np.var(delta_matrix, axis=0)  # [num_timesteps]
        
        # Select top-k timesteps with highest variance
        subset = np.argsort(variances)[-self.num_subset:]
        logger.info(f"Selected subset based on variance: {subset}")
        
        return subset