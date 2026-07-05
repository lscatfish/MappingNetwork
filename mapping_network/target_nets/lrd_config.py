from dataclasses import dataclass, field


def _normalize_enabled(value):
    """Normalize LRD enabled value to bool or 'auto'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ('true', '1', 'yes'):
            return True
        if lowered in ('false', '0', 'no'):
            return False
        if lowered in ('auto', ''):
            return 'auto'
    raise ValueError(f'Invalid LRD enabled value: {value!r}')


@dataclass
class LRDConfig:
    enabled: bool | str = 'auto'
    default_rank: int = 10
    layer_ranks: dict = field(default_factory=dict)
    layer_enabled: dict = field(default_factory=dict)
    auto_enable_threshold: int = 200_000

    def __post_init__(self):
        """Normalize string values like 'true' / 'false' to booleans."""
        self.enabled = _normalize_enabled(self.enabled)
        normalized = {}
        for key, value in self.layer_enabled.items():
            normalized[key] = _normalize_enabled(value)
        self.layer_enabled = normalized
