"""Resource configuration for the shared packed trainer; no script imports."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from cot_navigation.experiments.artifacts import digest

@dataclass(frozen=True)
class PackedBackend:
    here: Path
    base: Path
    protocols: tuple[str, ...]

    def source_hash(self, repo):
        repo = Path(repo).resolve()
        # Hash actual package code; study resource/config hashes are separate.
        paths = sorted((repo / 'src/cot_navigation').rglob('*.py'))
        return digest({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

def load_backend(name, repo, *, resources=None, world_root=None):
    repo = Path(repo).resolve()
    names = {'v6':'fixed_navigation_protocols_v6', 'v7':'fixed_navigation_hard_v7'}
    if name not in names:
        raise ValueError(f'Unknown packed backend: {name}')
    here = Path(resources).resolve() if resources else repo / 'studies' / names[name]
    base = Path(world_root).resolve() if world_root else (here / 'worlds' if name == 'v7' else repo / 'studies/fixed_navigation_v5')
    config = json.loads((here / 'experiment.json').read_text())
    return PackedBackend(here, base, tuple(config['protocols']))
