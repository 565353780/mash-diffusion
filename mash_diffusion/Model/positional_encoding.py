import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs * 2.0 * torch.pi
        x = torch.outer(x, freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
