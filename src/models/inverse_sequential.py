from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _has_class_name(module: nn.Module, class_name: str) -> bool:
    return module.__class__.__name__ == class_name


class InverseIdentity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class InverseFlatten(nn.Module):
    def __init__(self, output_shape: Sequence[int]):
        super().__init__()
        self.output_shape = tuple(output_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], *self.output_shape)


class InverseExtractCLS(nn.Module):
    """Expands the CLS token into a full sequence with trainable non-CLS tokens."""

    def __init__(self, seq_len: int, hidden_dim: int):
        super().__init__()
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_dim)
        if self.seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if self.seq_len > 1:
            self.non_cls_tokens = nn.Parameter(torch.zeros(1, self.seq_len - 1, self.hidden_dim))
            nn.init.normal_(self.non_cls_tokens, std=0.02)
        else:
            self.register_parameter("non_cls_tokens", None)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        if cls.dim() != 2:
            raise ValueError("InverseExtractCLS espera tensor [B, D]")
        b, d = cls.shape
        if d != self.hidden_dim:
            raise ValueError(f"Wrong CLS dimension: expected {self.hidden_dim}, got {d}")
        cls_token = cls.unsqueeze(1)
        if self.seq_len == 1:
            return cls_token
        rest = self.non_cls_tokens.expand(b, -1, -1).to(dtype=cls.dtype, device=cls.device)
        return torch.cat([cls_token, rest], dim=1)


class TrainableLayerNorm(nn.Module):
    """Does not invert LayerNorm: creates a new trainable LayerNorm."""

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.norm = nn.LayerNorm(ln.normalized_shape, eps=ln.eps, elementwise_affine=True)
        # Random weights for the normalisation, by design.
        nn.init.normal_(self.norm.weight, mean=1.0, std=0.02)
        nn.init.normal_(self.norm.bias, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class InverseLinearArchitecture(nn.Module):
    """Invierte arquitectura lineal: Linear(out_features -> in_features)."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.layer = nn.Linear(linear.out_features, linear.in_features, bias=True)
        self._init_from_minimum_norm_inverse(linear)

    def _init_from_minimum_norm_inverse(self, linear: nn.Linear) -> None:
        with torch.no_grad():
            # Forward: y = x W^T + b, W:[out,in]
            # Minimum-norm inverse: x_hat = (y - b) pinv(W)^T
            w = linear.weight.detach().float()
            w_pinv = torch.linalg.pinv(w).to(self.layer.weight.dtype)  # [in, out]
            self.layer.weight.copy_(w_pinv)

            if linear.bias is not None:
                b = linear.bias.detach().to(self.layer.weight.dtype)
                inv_b = -torch.matmul(b, w_pinv.T)
                self.layer.bias.copy_(inv_b)
            else:
                nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class InverseConvArchitecture(nn.Module):
    """Invierte arquitectura conv: Conv2d -> ConvTranspose2d entrenable."""

    def __init__(self, conv: nn.Conv2d, output_hw: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.layer = nn.ConvTranspose2d(
            in_channels=conv.out_channels,
            out_channels=conv.in_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        self._init_convtranspose(conv)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.kernel_size = conv.kernel_size
        self.output_hw = output_hw

    def _init_convtranspose(self, conv: nn.Conv2d) -> None:
        with torch.no_grad():
            # No exact inverse (expensive): stable initialisation for training.
            if (
                conv.in_channels == conv.out_channels
                and conv.groups == 1
                and conv.kernel_size[0] % 2 == 1
                and conv.kernel_size[1] % 2 == 1
            ):
                nn.init.dirac_(self.layer.weight)
            else:
                nn.init.kaiming_normal_(self.layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_padding = (0, 0)
        if self.output_hw is not None:
            h_in, w_in = x.shape[-2], x.shape[-1]
            base_h = (
                (h_in - 1) * self.stride[0]
                - 2 * self.padding[0]
                + self.dilation[0] * (self.kernel_size[0] - 1)
                + 1
            )
            base_w = (
                (w_in - 1) * self.stride[1]
                - 2 * self.padding[1]
                + self.dilation[1] * (self.kernel_size[1] - 1)
                + 1
            )
            out_pad_h = max(0, min(self.stride[0] - 1, self.output_hw[0] - base_h))
            out_pad_w = max(0, min(self.stride[1] - 1, self.output_hw[1] - base_w))
            output_padding = (out_pad_h, out_pad_w)

        return F.conv_transpose2d(
            x,
            self.layer.weight,
            self.layer.bias,
            stride=self.layer.stride,
            padding=self.layer.padding,
            output_padding=output_padding,
            groups=self.layer.groups,
            dilation=self.layer.dilation,
        )


class InversePoolUpsample(nn.Module):
    def __init__(self, output_hw: Tuple[int, int], mode: str = "nearest"):
        super().__init__()
        self.output_hw = output_hw
        self.mode = mode

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return F.interpolate(y, size=self.output_hw, mode=self.mode)


class InverseEncoderBlockArchitecture(nn.Module):
    """
    Trainable inverse architecture of a ViT EncoderBlock.

    Order of the inverse part:
      MLP -> MLP -> Norm -> AntiAttention -> Norm
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        hidden_dim = block.ln_1.normalized_shape[0]
        # Hidden dimension of the original MLP, to keep the scale
        mlp_dim = None
        for m in block.mlp.modules():
            if isinstance(m, nn.Linear):
                mlp_dim = m.out_features
                break
        if mlp_dim is None:
            mlp_dim = hidden_dim * 4

        self.mlp_inv_1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU()
        self.mlp_inv_2 = nn.Linear(mlp_dim, hidden_dim)
        self.norm_1 = nn.LayerNorm(hidden_dim)

        num_heads = block.self_attention.num_heads
        self.anti_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self._init_from_encoder_block(block)

    def _find_mlp_linears(self, block: nn.Module) -> List[nn.Linear]:
        return [m for m in block.mlp.modules() if isinstance(m, nn.Linear)]

    def _init_linear_as_min_norm_inverse(self, target: nn.Linear, source: nn.Linear) -> None:
        with torch.no_grad():
            w = source.weight.detach().float()
            w_pinv = torch.linalg.pinv(w).to(target.weight.dtype)
            target.weight.copy_(w_pinv)
            if source.bias is not None:
                b = source.bias.detach().to(target.weight.dtype)
                inv_b = -torch.matmul(b, w_pinv.T)
                target.bias.copy_(inv_b)
            else:
                nn.init.zeros_(target.bias)

    def _init_from_encoder_block(self, block: nn.Module) -> None:
        # Minimum-norm inverse MLP
        mlp_linears = self._find_mlp_linears(block)
        if len(mlp_linears) >= 2:
            # Reverse order: undo the second linear layer first, then the first one
            self._init_linear_as_min_norm_inverse(self.mlp_inv_1, mlp_linears[-1])
            self._init_linear_as_min_norm_inverse(self.mlp_inv_2, mlp_linears[0])
        else:
            nn.init.xavier_uniform_(self.mlp_inv_1.weight)
            nn.init.zeros_(self.mlp_inv_1.bias)
            nn.init.xavier_uniform_(self.mlp_inv_2.weight)
            nn.init.zeros_(self.mlp_inv_2.bias)

        # Random norms (by design)
        nn.init.normal_(self.norm_1.weight, mean=1.0, std=0.02)
        nn.init.normal_(self.norm_1.bias, mean=0.0, std=0.02)
        nn.init.normal_(self.norm_2.weight, mean=1.0, std=0.02)
        nn.init.normal_(self.norm_2.bias, mean=0.0, std=0.02)

        # Attention: initialised by copying the attention of the original block.
        src_attn = block.self_attention
        with torch.no_grad():
            if hasattr(src_attn, "in_proj_weight") and src_attn.in_proj_weight is not None:
                self.anti_attention.in_proj_weight.copy_(src_attn.in_proj_weight)
            else:
                nn.init.xavier_uniform_(self.anti_attention.in_proj_weight)

            if hasattr(src_attn, "in_proj_bias") and src_attn.in_proj_bias is not None:
                self.anti_attention.in_proj_bias.copy_(src_attn.in_proj_bias)
            else:
                nn.init.zeros_(self.anti_attention.in_proj_bias)

            self.anti_attention.out_proj.weight.copy_(src_attn.out_proj.weight)
            if src_attn.out_proj.bias is not None:
                self.anti_attention.out_proj.bias.copy_(src_attn.out_proj.bias)
            else:
                nn.init.zeros_(self.anti_attention.out_proj.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        x = self.mlp_inv_1(y)
        x = self.act(x)
        x = self.mlp_inv_2(x)
        x = self.norm_1(x)
        attn_out = self.anti_attention(x, x, x, need_weights=False)[0]
        x = x - attn_out
        x = self.norm_2(x)
        return x


class GenericTrainableMapper(nn.Module):
    """Fallback entrenable basado en forma de entrada/salida."""

    def __init__(self, input_shape: Sequence[int], output_shape: Sequence[int]):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.output_shape = tuple(output_shape)

        in_no_batch = self.input_shape[1:]
        out_no_batch = self.output_shape[1:]

        if len(in_no_batch) == 1 and len(out_no_batch) == 1:
            self.kind = "vec"
            self.mapper = nn.Linear(out_no_batch[0], in_no_batch[0], bias=True)
            nn.init.xavier_uniform_(self.mapper.weight)
            nn.init.zeros_(self.mapper.bias)

        elif len(in_no_batch) == 3 and len(out_no_batch) == 3:
            self.kind = "spatial"
            in_c, in_h, in_w = in_no_batch
            out_c, out_h, out_w = out_no_batch
            stride_h = max(1, in_h // max(1, out_h))
            stride_w = max(1, in_w // max(1, out_w))
            self.deconv = nn.ConvTranspose2d(
                out_c,
                in_c,
                kernel_size=3,
                stride=(stride_h, stride_w),
                padding=1,
                bias=True,
            )
            nn.init.kaiming_normal_(self.deconv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(self.deconv.bias)
            self.target_hw = (in_h, in_w)

        elif len(in_no_batch) == 2 and len(out_no_batch) == 2:
            self.kind = "token"
            in_t, in_d = in_no_batch
            out_t, out_d = out_no_batch
            self.token_proj = nn.Linear(out_d, in_d, bias=True)
            nn.init.xavier_uniform_(self.token_proj.weight)
            nn.init.zeros_(self.token_proj.bias)
            self.target_t = in_t
            self.out_t = out_t

        elif in_no_batch == out_no_batch:
            self.kind = "identity"
            self.mapper = InverseIdentity()
        else:
            raise ValueError(
                f"Cannot build a trainable fallback of shape {out_no_batch} -> {in_no_batch}."
            )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if self.kind == "vec":
            return self.mapper(y)
        if self.kind == "spatial":
            x = self.deconv(y)
            if x.shape[-2:] != self.target_hw:
                x = F.interpolate(x, size=self.target_hw, mode="bilinear", align_corners=False)
            return x
        if self.kind == "token":
            x = self.token_proj(y)
            if self.out_t != self.target_t:
                x = F.interpolate(
                    x.transpose(1, 2),
                    size=self.target_t,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            return x
        return self.mapper(y)


def _get_inverse_module(
    module: nn.Module,
    input_shape: Optional[Tuple[int, ...]],
    output_shape: Optional[Tuple[int, ...]],
    vit_fixed_point_steps: int,
    optimization_steps: int,
    optimization_lr: float,
) -> nn.Module:
    # Arguments kept for API compatibility
    _ = vit_fixed_point_steps
    _ = optimization_steps
    _ = optimization_lr

    if isinstance(module, (nn.Identity, nn.ReLU, nn.GELU, nn.SiLU, nn.Dropout, nn.BatchNorm2d)):
        return InverseIdentity()

    if isinstance(module, nn.Linear):
        return InverseLinearArchitecture(module)

    if isinstance(module, nn.Conv2d):
        if input_shape is None or len(input_shape) < 4:
            raise ValueError("Inverse Conv2d needs input_shape to rebuild the spatial size.")
        return InverseConvArchitecture(module, output_hw=(input_shape[-2], input_shape[-1]))

    if isinstance(module, nn.LayerNorm):
        return TrainableLayerNorm(module)

    if isinstance(module, nn.Flatten):
        if input_shape is None:
            raise ValueError("Flatten inversa necesita input_shape.")
        return InverseFlatten(output_shape=input_shape[1:])

    if isinstance(module, (nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
        if input_shape is None or len(input_shape) < 4:
            raise ValueError("Inverse pooling needs an input_shape with height and width.")
        return InversePoolUpsample(output_hw=(input_shape[-2], input_shape[-1]))

    if _has_class_name(module, "ExtractCLS"):
        if input_shape is None or len(input_shape) < 3:
            raise ValueError("ExtractCLS inversa necesita input_shape [B, seq_len, dim].")
        if output_shape is None or len(output_shape) < 2:
            raise ValueError("ExtractCLS inversa necesita output_shape [B, dim].")
        return InverseExtractCLS(seq_len=input_shape[1], hidden_dim=output_shape[-1])

    if _has_class_name(module, "EncoderBlock"):
        return InverseEncoderBlockArchitecture(module)

    if isinstance(module, nn.Sequential):
        if input_shape is None:
            raise ValueError("Inverting a nested nn.Sequential needs input_shape.")
        return invert_sequential_network(
            module,
            input_shape=input_shape[1:],
            vit_fixed_point_steps=vit_fixed_point_steps,
            optimization_steps=optimization_steps,
            optimization_lr=optimization_lr,
        )

    if input_shape is None:
        raise ValueError(
            f"No inverse implemented for {module.__class__.__name__} and no input_shape was given."
        )
    if output_shape is None:
        raise ValueError("No output_shape given for the generic fallback.")
    return GenericTrainableMapper(input_shape=input_shape, output_shape=output_shape)


def _infer_shapes(sequential: nn.Sequential, input_shape: Sequence[int]) -> List[Tuple[int, ...]]:
    with torch.no_grad():
        first_param = next(sequential.parameters(), None)
        if first_param is not None:
            x = torch.zeros(
                (1, *input_shape),
                dtype=first_param.dtype,
                device=first_param.device,
            )
        else:
            x = torch.zeros((1, *input_shape), dtype=torch.float32)
        shapes = [tuple(x.shape)]
        for module in sequential:
            x = module(x)
            shapes.append(tuple(x.shape))
    return shapes


def invert_sequential_network(
    sequential: nn.Sequential,
    input_shape: Optional[Sequence[int]] = None,
    vit_fixed_point_steps: int = 8,
    optimization_steps: int = 20,
    optimization_lr: float = 0.1,
) -> nn.Sequential:
    """
    Builds an (approximate) inverse network of an `nn.Sequential`.

    Args:
        sequential: forward network to invert.
        input_shape: shape of the original input WITHOUT the batch dimension (e.g. (197, 768)).
                     Needed for layers that destroy the shape (Flatten, Pool, ExtractCLS)
                     and for the optimisation fallback.
    """
    if input_shape is None:
        raise ValueError("input_shape (without batch) is required to build the inverse network.")

    shapes = _infer_shapes(sequential, input_shape)
    inverse_layers: List[nn.Module] = []

    n = len(sequential)
    for i in range(n - 1, -1, -1):
        module = sequential[i]
        in_shape = shapes[i]
        out_shape = shapes[i + 1]
        inverse_layers.append(
            _get_inverse_module(
                module,
                input_shape=in_shape,
                output_shape=out_shape,
                vit_fixed_point_steps=vit_fixed_point_steps,
                optimization_steps=optimization_steps,
                optimization_lr=optimization_lr,
            )
        )

    return nn.Sequential(*inverse_layers)
