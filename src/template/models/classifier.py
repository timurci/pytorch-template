from torch import Tensor, nn


class MLPClassifier(nn.Module):
    """Fully connected classifier over tabular features.

    A hidden block is ``Linear -> ReLU -> LayerNorm -> Dropout``. The model
    repeats ``hidden_depth`` blocks of ``hidden_size`` neurons, then a linear
    head with ``n_classes`` outputs. ``forward`` returns raw logits, so pair it
    with ``nn.CrossEntropyLoss``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_depth: int,
        n_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be at least 1")
        if hidden_depth < 0:
            raise ValueError("hidden_depth must be non-negative")
        if hidden_depth > 0 and hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        layers: list[nn.Module] = []
        previous = input_size
        for _ in range(hidden_depth):
            layers.append(nn.Linear(previous, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.Dropout(dropout))
            previous = hidden_size
        layers.append(nn.Linear(previous, n_classes))
        self._network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self._network(features)
