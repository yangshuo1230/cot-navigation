from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def change(rel, old, new):
    path = ROOT / rel
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f'{rel}: missing/ambiguous patch {old[:80]}')
    saved = HERE / 'before' / rel
    saved.parent.mkdir(parents=True, exist_ok=True)
    if not saved.exists():
        shutil.copy2(path, saved)
    path.write_text(text.replace(old, new))


change('src/cot_navigation/data/scale_prepared.py',
       '    return dict(source=source.fingerprint, protocol=protocol, vocab=tok.hash(), version=1,',
       '    return dict(**({"fixed_path_policy": "all_paths_v1"} if cfg.data.graph_family != "layered" and not isinstance(source, ContextStore) and protocol == "path" else {}),\n                source=source.fingerprint, protocol=protocol, vocab=tok.hash(), version=1,')
change('src/cot_navigation/data/scale_prepared.py',
       "        for variant in range(spec['variants']):",
       "        all_paths = spec.get('fixed_path_policy') and source.path_count(index) > 0\n        n_variants = source.path_count(index) if all_paths else spec['variants']\n        for variant in range(n_variants):")
change('src/cot_navigation/data/scale_prepared.py',
       "                selected = int(rng.integers(source.path_count(index))) if protocol == 'path' and source.path_count(index) else 0",
       "                selected = variant if all_paths else int(rng.integers(source.path_count(index))) if protocol == 'path' and source.path_count(index) else 0")
change('src/cot_navigation/data/scale_prepared.py',
       '            tokens.extend(row.tokens)',
       '            if cfg.data.graph_family != "layered" and row.meta.get("truncated", False):\n                raise ValueError(f"hard DAG reference exceeds context: {source.root} {protocol} query={index}; increase max_seq_len")\n            tokens.extend(row.tokens)')
change('src/cot_navigation/config.py',
       '    w = max(s.width for s in data.shapes.values())',
       '    w = max(s.width for s in data.shapes.values()) if data.graph_family == "layered" else n - 1')
change('src/cot_navigation/config.py',
       '        body = max(s.layers for s in cfg.data.shapes.values())',
       '        body = max(s.layers for s in cfg.data.shapes.values()) if cfg.data.graph_family == "layered" else cfg.data.n_nodes')
print('Hard DAG complete-path cache and context guards installed.')
