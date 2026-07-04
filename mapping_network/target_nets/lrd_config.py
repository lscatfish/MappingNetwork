from dataclasses import dataclass, field


@dataclass
class LRDConfig:
    enabled: bool | str = 'auto'
    default_rank: int = 10
    layer_ranks: dict = field(default_factory=dict)
    auto_enable_threshold: int = 200_000
