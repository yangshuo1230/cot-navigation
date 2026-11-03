"""Install opt-in hard DAG support; preserve all legacy default identities."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def change(rel, old, new, count=1):
    path = ROOT / rel
    text = path.read_text()
    if text.count(old) != count:
        raise RuntimeError(f'{rel}: expected {count} occurrences: {old[:100]!r}')
    saved = HERE / 'before' / rel
    saved.parent.mkdir(parents=True, exist_ok=True)
    if not saved.exists():
        shutil.copy2(path, saved)
    path.write_text(text.replace(old, new))


change('src/cot_navigation/config.py', '    node_vocab: int = 256',
       '    graph_family: str = "layered"  # layered | multiscale_dag_v1\n    node_vocab: int = 256')
change('src/cot_navigation/config.py', '        n_shapes = len(self.data.shapes)',
       '        _require(self.data.graph_family in {"layered", "multiscale_dag_v1"}, "graph_family")\n        n_shapes = len(self.data.shapes)')
change('src/cot_navigation/data/provenance.py', '    identity = {',
       '    if semantic.get("graph_family", "layered") == "layered":\n        semantic.pop("graph_family", None)\n    identity = {')
change('src/cot_navigation/data/graph.py', 'def sample_edges(n_layers: int, width: int, n_edges: int, rng: Generator)',
       'def sample_edges(n_layers: int, width: int, n_edges: int, rng: Generator, graph_family="layered")')
change('src/cot_navigation/data/graph.py', '    pool = candidate_edges(n_layers, width)',
       '    if graph_family == "multiscale_dag_v1":\n        from cot_navigation.data.hard_graph import sample_hard_edges\n        return sample_hard_edges(n_layers, width, n_edges, rng)\n    if graph_family != "layered":\n        raise ValueError(f"unknown graph family {graph_family}")\n    pool = candidate_edges(n_layers, width)')
change('src/cot_navigation/data/generate.py', '"n_edges": cfg.n_edges,',
       '"n_edges": cfg.n_edges,\n        "graph_family": cfg.graph_family,', count=2)
change('src/cot_navigation/data/generate.py', 'sample_edges(layers, width, n_edges, rng)',
       'sample_edges(layers, width, n_edges, rng, payload.get("graph_family", "layered"))')
change('src/cot_navigation/data/generate.py', 'sample_edges(layers, width, cfg.n_edges, rng)',
       'sample_edges(layers, width, cfg.n_edges, rng, cfg.graph_family)')
# With shortcuts, neighboring rank bands are NOT the same as one edge.
change('src/cot_navigation/data/generate.py', 'rows.extend((0, 1, int(v in adj[u]), u, v)',
       'rows.extend((0, 1, int(reaches(reach, u, v)) if cfg.graph_family != "layered" else int(v in adj[u]), u, v)')
change('src/cot_navigation/data/generate.py', '        if distance >= 5 and seen:',
       '        if cfg.graph_family == "layered" and distance >= 5 and seen:')
change('src/cot_navigation/data/validate.py', '\n        if u // spec.width + 1 != v // spec.width:',
       '\n        if ((u // spec.width + 1 != v // spec.width) if cfg.graph_family == "layered" else not u < v):')
change('src/cot_navigation/data/validate.py', '\n            if u // spec.width + 1 != v // spec.width:',
       '\n            if ((u // spec.width + 1 != v // spec.width) if cfg.graph_family == "layered" else not u < v):')
change('src/cot_navigation/data/validate.py', '        path_stats = _validate_paths(world, adj)',
       '        path_stats = _validate_paths(world, adj, cfg.graph_family == "layered")')
change('src/cot_navigation/data/validate.py', '            if distance >= 5 and int(world.queries["witness_seen"][i]) == 1:',
       '            if cfg.graph_family == "layered" and distance >= 5 and int(world.queries["witness_seen"][i]) == 1:')
change('src/cot_navigation/data/validate.py', 'def _validate_paths(world: ParametricWorld, adj: list[list[int]])',
       'def _validate_paths(world: ParametricWorld, adj: list[list[int]], layered=True)')
change('src/cot_navigation/data/validate.py', '                    or len(path) != int(world.queries["distance"][i]) + 1',
       '                    or (layered and len(path) != int(world.queries["distance"][i]) + 1)')
shutil.copy2(HERE / 'hard_graph.py', ROOT / 'src/cot_navigation/data/hard_graph.py')
print('Opt-in hard DAG support installed.')
