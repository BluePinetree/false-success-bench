"""Early-stopping target check."""
from .metrics import to_percent


def target_reached(val_metrics, target):
    """True once the tracked validation metric meets the configured target."""
    name = target["metric"].removeprefix("val_")
    return to_percent(val_metrics[name]) >= target["value"]
