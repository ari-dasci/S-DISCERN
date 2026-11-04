import torch
import torch.nn as nn
import torch.nn.functional as F

class ADINGenerativeModel(nn.Module):
    def __init__(self, model : nn.Sequential, noisy_dim : int):
        super().__init__()
        self.model = model
        self.noisy_dim = noisy_dim
        self.noisy_upsampler = nn.ModuleDict()
        for i in range(len(self.model)):
            if isinstance(self.model[i], torch.nn.LayerNorm):
                norm_shape = self.model[i].normalized_shape
                if isinstance(norm_shape, int):
                    norm_shape = (norm_shape,)
                num_features = 1
                for dim in norm_shape:
                    num_features *= int(dim)
                self.noisy_upsampler[str(i)] = torch.nn.Linear(
                    noisy_dim,
                    num_features * 2,
                    bias=True,
                )

    def forward(self, x, w):
        for i, layer in enumerate(self.model):
            if isinstance(layer, torch.nn.LayerNorm):
                norm_shape = layer.normalized_shape
                if isinstance(norm_shape, int):
                    norm_shape = (norm_shape,)
                noisy_emb = self.noisy_upsampler[str(i)](w)
                style = noisy_emb.view(noisy_emb.size(0), 2, *norm_shape)

                gamma = style[:, 0] # Escalas
                beta = style[:, 1]  # Desplazamientos
                x = layer(x) * (gamma + 1) + beta
            else:
                x = layer(x)

        return x
