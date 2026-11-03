from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
"""Compatibility adapter for the shared packed corpus implementation."""
from prepare import HERE, BASE
from cot_navigation.data.packed_bank import PackedBank as _PackedBank
from cot_navigation.training.packed_loss import logical_loss

class PackedBank(_PackedBank):
    def __init__(self, meta, protocol, device, verify_hash=True):
        super().__init__(meta, protocol, device, verify_hash,
                         bank_root=HERE, world_root=BASE)
