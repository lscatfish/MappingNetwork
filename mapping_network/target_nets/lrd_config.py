from dataclasses import dataclass, field


@dataclass
class LRDConfig:
    enabled: bool | str = 'auto'
    default_rank: int = 10
    layer_ranks: dict = field(default_factory=dict)
    auto_enable_threshold: int = 200_000

    def __post_init__(self):
        """Normalize string values like 'true' / 'false' to booleans."""
        if isinstance(self.enabled, str):
            lowered = self.enabled.lower()
            if lowered in ('true', '1', 'yes'):
                self.enabled = True
            elif lowered in ('false', '0', 'no'):
                self.enabled = False
            elif lowered in ('auto', ''):
                self.enabled = 'auto'
            else:
                raise ValueError(f'Invalid LRDConfig.enabled value: {self.enabled!r}')
