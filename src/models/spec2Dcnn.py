from typing import Optional

import torch
import torch.nn as nn

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup


class Spec2DCNN(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        encoder: nn.Module,
        loss_type,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.decoder = decoder
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)

        self.losses_objs = {
            "mse": nn.MSELoss(),
            "bce": nn.BCEWithLogitsLoss(),
            "kld": nn.KLDivLoss()
        }

        self.loss_types = loss_type.split("+")

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_timesteps, n_classes)
        Returns:
            dict[str, torch.Tensor]: logits (batch_size, n_timesteps, n_classes)
        """
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        x = self.encoder(x).squeeze(1)  # (batch_size, height, n_timesteps)
        logits = self.decoder(x)  # (batch_size, n_classes, n_timesteps)

        output = {"logits": logits}
        if labels is not None:
            loss = None
            for loss_type in self.loss_types:
                if loss is None:
                    loss = self.losses_objs[loss_type](logits, labels)
                else:
                    loss = loss + self.losses_objs[loss_type](logits, labels)
            output["loss"] = loss

        return output
