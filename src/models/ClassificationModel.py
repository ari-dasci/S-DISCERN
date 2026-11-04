import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, Swin_T_Weights
from .inverse_sequential import invert_sequential_network

import numpy as np

class ExtractCLS(nn.Module):
    def forward(self, x):
        # Keep only the first token of the sequence
        return x[:, 0]


class TemperatureScaling(nn.Module):
    """Post-classifier learnable temperature: logits → logits / T.

    Initialized to T=1 (identity) with requires_grad=False.
    Enable training via requires_grad_(True) from outside.
    Penalise T < 0.1 externally to prevent logit sharpening collapse.

    ``active`` (buffer, default True) gates whether ``forward`` actually
    applies the scaling. When False, forward is the identity regardless of
    ``temperature`` — used by forget-temperature calibration (see
    DistributionalMethod._calibrate_forget_temperature /
    --forget-temperature-always-apply) to store a calibrated temperature
    without necessarily baking it into every future inference call.

    ``inference_only`` (buffer, default False) additionally bypasses scaling
    whenever the module is in training mode (``self.training``), applying it
    only in eval mode — see --temperature-inference-only. This lets the
    forget-temperature-calibration correction affect evaluation/inference
    without perturbing the training-time logits (e.g. the synthetic
    conservative loss, notop, confidence-penalty terms).
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)
        self.register_buffer("active", torch.tensor(True))
        self.register_buffer("inference_only", torch.tensor(False))

    def forward(self, x):
        if not bool(self.active):
            return x
        if bool(self.inference_only) and self.training:
            return x
        return x / self.temperature.clamp(min=1e-6)


class ClassificationModel(nn.Module):
    def __init__(self, base_model, num_classes, hyperparameters, classes_weights=None, extra_layers=True, pretrained=True):
        super(ClassificationModel, self).__init__()
        # Default hyperparameters
        self._hyperparameters = hyperparameters
        self._num_classes = num_classes
        self._base_model = base_model
        self._extra_layers = extra_layers

        if self._base_model == "resnet18":

            self._model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

            if not hasattr(self._model, "fc") or not isinstance(self._model.fc, nn.Sequential):
                in_features = self._model.fc.in_features
                if extra_layers == True:
                    self._model.fc = nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(in_features=in_features, out_features=self._hyperparameters['embedding_size'], bias=True)
                    )

                else:
                    self._model.fc = nn.Identity()
                    self._hyperparameters['embedding_size'] = in_features
        
        elif self._base_model == "efficientnet_v2_s":
            # Load backbone
            self._model = torch.hub.load("pytorch/vision:v0.16.2", "efficientnet_v2_s" , weights='IMAGENET1K_V1' if pretrained else None)

            # Set the new layers
            in_features = self._model.features[-1].out_channels

            if extra_layers == True:
                self._model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_features=in_features, out_features=self._hyperparameters['embedding_size'], bias=True)
                )
            else:
                self._model.classifier = nn.Identity()
                self._hyperparameters['embedding_size'] = in_features
        
        elif self._base_model == "vit_b_16":
            # Load backbone
            self._model = torch.hub.load("pytorch/vision:v0.16.2", "vit_b_16", weights='IMAGENET1K_V1' if pretrained else None)

            # Set the new layers
            in_features = self._model.heads[0].in_features

            if extra_layers == True:
                self._model.heads = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_features=in_features, out_features=self._hyperparameters['embedding_size'], bias=True)
                )

            else:
                self._model.heads = nn.Identity()
                self._hyperparameters['embedding_size'] = in_features

        elif self._base_model == "swin_t":
            self._model = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)

            in_features = self._model.head.in_features  # 768

            if extra_layers == True:
                self._model.head = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_features=in_features, out_features=self._hyperparameters['embedding_size'], bias=True)
                )
            else:
                self._model.head = nn.Identity()
                self._hyperparameters['embedding_size'] = in_features

        else:
            raise ValueError("Invalid base model! Choose 'resnet18', 'efficientnet_v2_s', 'vit_b_16' or 'swin_t'.")
        
        self.classifier = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(in_features=self._hyperparameters['embedding_size'], out_features=num_classes, bias=True),
            TemperatureScaling(),
        )

        if classes_weights is not None:
            self.classes_weights = classes_weights
        else:
            self.classes_weights = np.ones(num_classes, dtype=np.float32)

        self.criterion = nn.CrossEntropyLoss(torch.from_numpy(self.classes_weights).float())

    @property
    def device(self):
        """Return the device of the model's first parameter."""
        return next(self.parameters()).device

    def forward(self, images, return_feature=False, return_only_embs=False):
        embeddings = self._model(images)
        if return_only_embs:
            return embeddings.float()
        logits = self.classifier(embeddings)

        if return_feature:
            return logits.float(), embeddings.float()
        else:
            return logits.float()

    def freeze(self):
        for name, param in self._model.named_parameters():
            param.requires_grad = False
        
        for name, param in self.classifier.named_parameters():
            param.requires_grad = True
        self.classifier[-1].temperature.requires_grad_(False)

    # Backbone parameters trained in "embs" mode without extra_layers: the last
    # stage of each architecture (names relative to self._model).  Shared by
    # emb_freeze() and the methods that edit weights in place (SSD,
    # FISHERFORGETTING) through is_emb_param().
    _EMB_TRAINABLE_PREFIXES = {
        "resnet18": ("layer4.",),                                            # last stage (2 residual blocks)
        "efficientnet_v2_s": ("features.6.", "features.7."),                 # last MBConv stage + final 1x1 conv
        "vit_b_16": ("encoder.layers.encoder_layer_11.", "encoder.ln."),     # last block + final LayerNorm
        "swin_t": ("features.7.", "norm."),                                  # last stage + final LayerNorm
    }
    _EXTRA_LAYER_HEADS = {"resnet18": "fc", "efficientnet_v2_s": "classifier",
                          "vit_b_16": "heads", "swin_t": "head"}

    def is_emb_param(self, name):
        """True if the parameter ``name`` (as given by self.named_parameters())
        is trained in "embs" mode: the classifier, plus the extra embedding layer
        (extra_layers=True) or the last backbone stage (_EMB_TRAINABLE_PREFIXES)."""
        if name.startswith("classifier."):
            return True
        if not name.startswith("_model."):
            return False
        name = name[len("_model."):]
        if self._extra_layers == True:
            return self._EXTRA_LAYER_HEADS[self._base_model] in name
        return name.startswith(self._EMB_TRAINABLE_PREFIXES[self._base_model])

    def emb_freeze(self):
        """
        Freezes the whole backbone (every convolutional / attention layer),
        keeping trainable only:
        - the embedding layer (if extra_layers=True)
        - the final classifier

        Without extra_layers, only the LAST stage of each architecture is unfrozen
        (_EMB_TRAINABLE_PREFIXES):
        - resnet18: layer4 (its 2 residual blocks)
        - efficientnet_v2_s: features.6 (last MBConv stage) + features.7 (final 1x1 conv)
        - vit_b_16: encoder.layers.encoder_layer_11 + encoder.ln (last attention layer + final layer norm)
        - swin_t: features.7 + norm (last stage + final layer norm)

        Works for every architecture, with extra_layers=True or False.
        """
        for name, param in self._model.named_parameters():
            if self._extra_layers == True:
                # With extra_layers, freeze everything except the embedding layer
                if self._base_model == "resnet18":
                    if "fc" not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                elif self._base_model == "efficientnet_v2_s":
                    if "classifier" not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                elif self._base_model == "vit_b_16":
                    if "heads" not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                elif self._base_model == "swin_t":
                    if "head" not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            else:
                # Without extra_layers, unfreeze only the LAST stage of the backbone.
                # (Before Oct 2026 efficientnet_v2_s used `"features" in name`, i.e. the
                # WHOLE backbone: those runs were renamed to *_nofreeze.)
                param.requires_grad = name.startswith(self._EMB_TRAINABLE_PREFIXES[self._base_model])

        # Keep the final classifier trainable
        for name, param in self.classifier.named_parameters():
            param.requires_grad = True
        self.classifier[-1].temperature.requires_grad_(False)


    def get_fc(self):
        return self.classifier[-2]

    def _add_logits(self, num_add):
        """Expand classifier output dimension while preserving learned logits."""
        if num_add <= 0:
            return

        old_fc = self.get_fc()
        in_features = old_fc.in_features
        old_out_features = old_fc.out_features
        new_out_features = old_out_features + num_add

        new_fc = nn.Linear(in_features=in_features, out_features=new_out_features, bias=True)
        with torch.no_grad():
            new_fc.weight[:old_out_features].copy_(old_fc.weight)
            new_fc.bias[:old_out_features].copy_(old_fc.bias)

        self.classifier[-2] = new_fc
        self._num_classes = new_out_features
        self.classes_weights = np.ones(new_out_features, dtype=np.float32)
        self.criterion = nn.CrossEntropyLoss(torch.from_numpy(self.classes_weights).float())

    def get_embs_layer(self):
        if not self._extra_layers:
            # Return the last body layer of each architecture
            if self._base_model == "resnet18":
                return nn.Sequential(
                            self._model.layer4,          # The last convolutions (the "extractor")
                            self._model.avgpool,         # Dimensionality reduction (analogous to the slicing)
                            nn.Flatten(1)          # From (B, 2048, 1, 1) to (B, 2048)
                        )
            elif self._base_model == "efficientnet_v2_s":
                return nn.Sequential(
                            self._model.features[-1], # The 1x1 conv
                            self._model.avgpool,      # The pooling
                            nn.Flatten(1)             # The 1280-d vector
                        )
            elif self._base_model == "vit_b_16":
                return nn.Sequential(
                            self._model.encoder.layers.encoder_layer_11, # The last Transformer block
                            self._model.encoder.ln,                      # The final normalisation (REQUIRED)
                            ExtractCLS()                                 # Automatic slicing
                        )
            elif self._base_model == "swin_t":
                # features[7] takes (B, 7, 7, 768) and returns (B, 7, 7, 768)
                # norm + permute + avgpool + flatten -> (B, 768)
                return nn.Sequential(
                            self._model.features[7],
                            self._model.norm,
                            self._model.permute,
                            self._model.avgpool,
                            self._model.flatten,
                        )
            else:
                raise ValueError("Invalid base model!")
        
        if self._base_model == "resnet18":
            return self._model.fc[-1]  # nn.Sequential(Dropout, Linear) -> Linear
        elif self._base_model == "efficientnet_v2_s":
            return self._model.classifier[-1]  # nn.Sequential(Dropout, Linear) -> Linear
        elif self._base_model == "vit_b_16":
            return self._model.heads[-1]  # nn.Sequential(Dropout, Linear) -> Linear
        elif self._base_model == "swin_t":
            return self._model.head[-1]   # nn.Sequential(Dropout, Linear) -> Linear
        else:
            raise ValueError("Invalid base model! Choose 'resnet18', 'efficientnet_v2_s', 'vit_b_16' or 'swin_t'.")
        
    def get_input_dim_of_sequential(self):
        if self._base_model == "resnet18":
            # The input of layer4 comes from layer3,
            # but it is easier to read the input channels of the first block of layer4
            return self._model.layer4[0].conv1.in_channels

        elif self._base_model == "efficientnet_v2_s":
            # The input of features[-1] (the 1x1 conv)
            return self._model.features[-1][0].in_channels

        elif self._base_model == "vit_b_16":
            # In the ViT every attention layer keeps the same dimension
            return self._model.hidden_dim

        elif self._base_model == "swin_t":
            # PatchMerging before features[7] outputs 768 channels (HWC layout)
            return 768

    def get_inverse_sequential(self, sequential_layer, input_shape, vit_fixed_point_steps=8, optimization_steps=20, optimization_lr=0.1):
        """
        Builds the trainable inverse architecture of any nn.Sequential.

        Args:
            sequential_layer: nn.Sequential layer to invert.
            input_shape: input shape of `sequential_layer`, WITHOUT the batch dimension.
        """
        if not isinstance(sequential_layer, nn.Sequential):
            raise ValueError("sequential_layer must be an nn.Sequential")

        return invert_sequential_network(
            sequential_layer,
            input_shape=input_shape,
            vit_fixed_point_steps=vit_fixed_point_steps,
            optimization_steps=optimization_steps,
            optimization_lr=optimization_lr,
        )

    def get_inverse_embs_layer(self, input_shape, vit_fixed_point_steps=8, optimization_steps=20, optimization_lr=0.1):
        """
        Returns the trainable inverse architecture of `get_embs_layer()`.
        """
        embs_layer = self.get_embs_layer()
        if not isinstance(embs_layer, nn.Sequential):
            raise ValueError("get_embs_layer() must return an nn.Sequential to be inverted")

        return invert_sequential_network(
            embs_layer,
            input_shape=input_shape,
            vit_fixed_point_steps=vit_fixed_point_steps,
            optimization_steps=optimization_steps,
            optimization_lr=optimization_lr,
        )