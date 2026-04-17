"""
Simplified State Space Model (SSM) Block
-----------------------------------------
Approximates the Mamba/S4 selective scan mechanism using:
  - Depthwise convolution for local feature extraction
  - Lightweight selective gating for global context modeling
  - Linear complexity O(N) in sequence length

This avoids requiring CUDA-specific mamba-ssm kernels while preserving
the architecture's intent of linear-complexity global context modeling.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SSMKernel(nn.Module):
    """
    Simplified selective SSM kernel via efficient 1-D convolution + gating.
    Processes spatial sequences (H*W) with linear complexity.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = int(d_model * expand)
        self.d_inner = d_inner

        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # SSM parameters (selective: input-dependent Δ, B, C)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # Depthwise conv for local mixing (mimics local scan)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=True
        )

        # Learnable A (log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(d_inner, -1)  # d_inner x d_state
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # Normalization
        self.norm = nn.LayerNorm(d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)   where L = H*W (flattened spatial)
        returns: (B, L, d_model)
        """
        B, L, _ = x.shape

        # Gate split
        xz = self.in_proj(x)               # B L (2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)      # B L d_inner each

        # Local conv pass (transpose to B d_inner L for Conv1d)
        x_conv = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Selective SSM parameters
        delta_BC = self.x_proj(x_conv)           # B L (2*d_state+1)
        delta, B_mat, C_mat = delta_BC.split(
            [1, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))   # B L d_inner

        A = -torch.exp(self.A_log.float())        # d_inner x d_state

        # Efficient scan via cumulative product approximation
        # (True Mamba uses parallel selective scan; here we use a simpler
        #  but differentiable surrogate via cumsum + softmax attention)
        y = self._selective_scan_approx(x_conv, delta, A, B_mat, C_mat)

        # Skip connection (D term)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # Gating
        y = y * F.silu(z)
        y = self.norm(y)
        return self.out_proj(y)

    def _selective_scan_approx(
        self,
        u: torch.Tensor,      # B L d_inner
        delta: torch.Tensor,  # B L d_inner
        A: torch.Tensor,      # d_inner d_state
        B: torch.Tensor,      # B L d_state
        C: torch.Tensor,      # B L d_state
    ) -> torch.Tensor:
        """
        Memory-efficient selective scan approximation via chunked parallel scan.
        Processes sequence in CHUNK_SIZE steps per iteration to bound memory usage.
        This gives O(L / chunk * chunk) = O(L) time with bounded peak memory.
        """
        B_sz, L, d_inner = u.shape
        d_state = A.shape[-1]
        CHUNK = 64   # process 64 timesteps at a time

        # Decay factors (scalar per step per dimension, no full L×d_inner×d_state tensor)
        # A: (d_inner, d_state)  →  log_decay per dimension
        log_A = A  # already negative (set in __init__ as -exp(A_log))

        output = torch.zeros(B_sz, L, d_inner, device=u.device, dtype=u.dtype)
        h = torch.zeros(B_sz, d_inner, d_state, device=u.device, dtype=u.dtype)

        for start in range(0, L, CHUNK):
            end = min(start + CHUNK, L)
            T = end - start

            # Slice chunk
            u_c     = u[:, start:end, :]          # B T d_inner
            delta_c = delta[:, start:end, :]      # B T d_inner
            B_c     = B[:, start:end, :]          # B T d_state
            C_c     = C[:, start:end, :]          # B T d_state

            # dA_c: (B, T, d_inner, d_state)
            dA_c = torch.exp(
                delta_c.unsqueeze(-1) * log_A.unsqueeze(0).unsqueeze(0)
            )
            # dB_c: (B, T, d_inner, d_state)
            dB_c = delta_c.unsqueeze(-1) * B_c.unsqueeze(2)
            x_c = dB_c * u_c.unsqueeze(-1)

            # Parallel prefix-scan over affine recurrences:
            #   h_t = dA_t * h_{t-1} + x_t
            # Compose transforms in O(log T) scan stages instead of a Python
            # loop over timesteps, while avoiding division by tiny prefix
            # products that can create inf/nan values.
            a_prefix = dA_c.clone()
            b_prefix = x_c.clone()
            offset = 1
            while offset < T:
                next_a = a_prefix.clone()
                next_b = b_prefix.clone()
                a_right = a_prefix[:, offset:]
                b_right = b_prefix[:, offset:]
                a_left = a_prefix[:, :-offset]
                b_left = b_prefix[:, :-offset]

                next_a[:, offset:] = a_right * a_left
                next_b[:, offset:] = b_right + a_right * b_left
                a_prefix = next_a
                b_prefix = next_b
                offset *= 2

            h_all = a_prefix * h.unsqueeze(1) + b_prefix

            y_c = (h_all * C_c.unsqueeze(2)).sum(-1)
            output[:, start:end, :] = y_c
            h = h_all[:, -1]

        return output  # (B, L, d_inner)


class MambaBlock2D(nn.Module):
    """
    2-D Mamba-style block operating on spatial feature maps.
    For memory efficiency, the SSM operates on a STRIDED subgrid then
    upsamples back – capturing global context while bounding memory.
    """

    def __init__(self, channels: int, d_state: int = 16, expand: int = 2,
                 max_seq_len: int = 256):
        super().__init__()
        self.channels    = channels
        self.max_seq_len = max_seq_len   # max spatial positions for SSM
        self.norm1 = nn.LayerNorm(channels)
        self.ssm   = SSMKernel(channels, d_state, expand)

        # Lightweight positional mixing via depthwise conv (fallback for large maps)
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels, bias=True
        )
        self.gate = nn.Parameter(torch.ones(1))

    @staticmethod
    def _largest_divisor_at_most(value: int, limit: int) -> int:
        """
        Return the largest divisor of ``value`` that does not exceed ``limit``.

        This is used to keep MPS downsampling on exact integer pooling windows,
        because adaptive average pooling on Metal only supports divisible sizes.
        """
        limit = max(1, min(int(limit), int(value)))
        for candidate in range(limit, 0, -1):
            if value % candidate == 0:
                return candidate
        return 1

    def _downsample_for_ssm(
        self,
        x: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> tuple[torch.Tensor, int, int]:
        """
        Downsample feature maps for the global SSM branch.

        On MPS, avoid adaptive pooling for non-divisible target sizes by falling
        back to exact average pooling with divisor-compatible output shapes.
        """
        _, _, H, W = x.shape
        target_h = max(1, min(int(target_h), H))
        target_w = max(1, min(int(target_w), W))

        if x.device.type == "mps":
            target_h = self._largest_divisor_at_most(H, target_h)
            target_w = self._largest_divisor_at_most(W, target_w)
            kernel_h = max(1, H // target_h)
            kernel_w = max(1, W // target_w)
            x_small = F.avg_pool2d(
                x,
                kernel_size=(kernel_h, kernel_w),
                stride=(kernel_h, kernel_w),
            )
            return x_small, target_h, target_w

        x_small = F.adaptive_avg_pool2d(x, (target_h, target_w))
        return x_small, target_h, target_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        L = H * W

        # Local mixing always applied
        local = self.dw_conv(x)   # (B, C, H, W)

        # Global SSM only if sequence is manageable
        if L <= self.max_seq_len:
            x_flat = rearrange(x, 'b c h w -> b (h w) c')
            ssm_out = self.ssm(self.norm1(x_flat))   # (B, L, C)
            ssm_out = rearrange(ssm_out, 'b (h w) c -> b c h w', h=H, w=W)
        else:
            # Downsample → SSM → upsample (approximate global context)
            scale = (self.max_seq_len / L) ** 0.5
            h_s = max(1, int(H * scale))
            w_s = max(1, int(W * scale))
            x_small, h_s, w_s = self._downsample_for_ssm(x, h_s, w_s)
            x_flat   = rearrange(x_small, 'b c h w -> b (h w) c')
            ssm_small = self.ssm(self.norm1(x_flat))          # (B, h_s*w_s, C)
            ssm_small = rearrange(ssm_small, 'b (h w) c -> b c h w', h=h_s, w=w_s)
            ssm_out   = F.interpolate(ssm_small, size=(H, W), mode='bilinear',
                                      align_corners=False)

        # Gate-weighted fusion of local + global
        gate = torch.sigmoid(self.gate)
        return x + gate * ssm_out + (1 - gate) * local


class CNNSSMBlock(nn.Module):
    """
    CNN-SSM hybrid block (core of U-Mamba encoder/decoder).
    Local feature extraction via CNN + global context via SSM.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        d_state:      int = 16,
        expand:       int = 2,
        dropout:      float = 0.1,
    ):
        super().__init__()

        # CNN branch (local)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Channel adapter if needed
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.skip_conv = nn.Identity()

        # SSM branch (global)
        self.ssm = MambaBlock2D(out_channels, d_state=d_state, expand=expand)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_conv(x)
        feat = self.cnn(x)
        feat = feat + residual
        feat = self.ssm(feat)
        return self.dropout(feat)
