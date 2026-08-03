"""CNN architecture.

Four convolutional blocks followed by global average pooling. GAP (rather than
flatten) keeps the parameter count at ~60k and makes the network accept inputs
of any time length, which the streaming path relies on.
"""

import torch.nn as nn

_CHANNELS = (1, 16, 32, 64, 64)


def _block(in_ch: int, out_ch: int) -> list[nn.Module]:
    """One conv block, returned flat.

    The layers are spliced into a single ``nn.Sequential`` rather than nested in
    a sub-module: nesting would rename every state_dict key and break existing
    checkpoints.
    """
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    ]


class GenderCNN(nn.Module):
    """Log-mel spectrogram -> 2 logits.

    Input:  (batch, 1, n_mels, n_frames)
    Output: (batch, n_classes) raw logits — apply softmax yourself if you need
            probabilities. Do not add a softmax layer here; ``CrossEntropyLoss``
            expects logits.
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            *[
                layer
                for in_ch, out_ch in zip(_CHANNELS, _CHANNELS[1:])
                for layer in _block(in_ch, out_ch)
            ]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(_CHANNELS[-1], n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
