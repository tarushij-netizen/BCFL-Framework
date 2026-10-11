"""
Data-poisoning attacks (the malicious client trains on poisoned data, then uploads
normally). These match the attacks Leash-FL evaluates:

  labelflip : every label y becomes (num_classes - 1 - y)
  backdoor  : in each batch, a fraction of images gets a small white square (trigger)
              in the bottom-right corner and its label is set to `target`.
              Success is measured as ASR: the share of triggered test images (true
              label != target) that the global model classifies as `target`.
"""
DATA_ATTACKS = ("labelflip", "backdoor")


def add_trigger(x, size=3, value=1.0):
    """White size x size square, 1 pixel in from the bottom-right corner. x: (B, C, H, W)."""
    x = x.clone()
    x[..., -size - 1:-1, -size - 1:-1] = value
    return x


class PoisonedLoader:
    """Wraps a DataLoader and poisons each batch on the fly. Supports iteration and len()."""

    def __init__(self, loader, attack, num_classes=10, target=0, poison_frac=0.5):
        if attack not in DATA_ATTACKS:
            raise ValueError(f"Unknown data attack '{attack}' (use {DATA_ATTACKS})")
        self.loader = loader
        self.attack = attack
        self.num_classes = num_classes
        self.target = target
        self.poison_frac = poison_frac

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for x, y in self.loader:
            if self.attack == "labelflip":
                yield x, (self.num_classes - 1) - y
            else:
                k = int(round(self.poison_frac * len(y)))
                if k > 0:
                    x = x.clone()
                    y = y.clone()
                    x[:k] = add_trigger(x[:k])
                    y[:k] = self.target
                yield x, y
