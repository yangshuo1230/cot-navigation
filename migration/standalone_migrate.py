"""Migrate only the independent project; never mutate the source repository."""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil
import subprocess

ROOT=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
ARCHIVE=ROOT/'research'
OUT=ROOT/'migration/standalone'
OUT.mkdir(parents=True,exist_ok=True)

def write(rel,text):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)

def extract(path,names):
    text=path.read_text();lines=text.splitlines(True);tree=ast.parse(text)
    found={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    return '\n\n'.join(''.join(lines[found[name].lineno-1:found[name].end_lineno]) for name in names)

def replace_definitions(path,names,imports):
    text=path.read_text();lines=text.splitlines(True);tree=ast.parse(text)
    ranges=[(n.lineno-1,n.end_lineno) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    for start,end in sorted(ranges,reverse=True):
        lines[start:end]=['\n']
    tree=ast.parse(''.join(lines))
    # Insert after future imports and module docstring.
    at=0
    for node in tree.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            at=node.end_lineno
        elif isinstance(node,ast.ImportFrom) and node.module=='__future__':
            at=node.end_lineno
        else:break
    lines.insert(at,imports+'\n')
    path.write_text(''.join(lines))

# Capture a local numeric oracle once; tests must never read the old repository.
oracle=ROOT.with_name('cot-navigation')/'research/fixed_navigation_protocols_v6_20261026/packed_bank.py'
write('tests/fixtures/packed_reference.py.txt',oracle.read_text())

mapping={p.name:re.sub(r'_\d{8}$','',p.name) for p in ARCHIVE.iterdir() if p.is_dir() and p.name!='__pycache__'}
assert len(set(mapping.values()))==len(mapping),'Study identifiers collide'
(OUT/'study_paths.json').write_text(json.dumps(mapping,indent=2))
studies=ROOT/'studies';studies.mkdir(exist_ok=True)
for source in ARCHIVE.iterdir():
    if source.name=='__pycache__':continue
    target=studies/mapping.get(source.name,source.name)
    if target.exists():raise FileExistsError(target)
    subprocess.run(['cp','-a','--reflink=auto',str(source),str(target)],check=True)
write('studies/__init__.py','"""Self-contained research applications and their frozen study resources."""\n')

# Relocate active code, shell launchers and config references. Archived results
# remain byte-for-byte unchanged; only executable/configuration text is edited.
changed=[]
files=[]
for folder in ('src','scripts','studies','tests','configs'):
    files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.sh','.yaml','.yml'} and '__pycache__' not in p.parts]
for p in files:
    original=p.read_text()
    text=original
    # Names are replaced only inside strings: identifiers are left untouched.
    if p.suffix=='.py':
        import io,tokenize
        tokens=[]
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type==tokenize.STRING:
                value=tok.string
                for old,new in mapping.items():value=value.replace(old,new)
                value=value.replace('research/','studies/').replace('/research','/studies')
                if value in {"'research'",'"research"'} and p.name!='workflow.py':
                    value=value.replace('research','studies')
                tok=tok._replace(string=value)
            tokens.append(tok)
        text=tokenize.untokenize(tokens)
    else:
        for old,new in mapping.items():text=text.replace(old,new)
        text=text.replace('research/','studies/')
    # Keep archive provenance/testing policies out of generic replacement.
    if text!=original:
        p.write_text(text);changed.append(str(p.relative_to(ROOT)))

# Shared runtime components replace version-script loading.
adj=studies/mapping['fixed_adjacency_suite_20261024']/'train_suite.py'
write('src/cot_navigation/data/adjacency.py','"""Query/adjacency sample encoding and exact token-budget packing."""\n'+extract(adj,['Samples'])+'\n')
write('src/cot_navigation/evaluation/adjacency.py','"""Autoregressive full adjacency-list evaluation."""\nfrom collections import Counter\n'+extract(adj,['evaluate_adjacency'])+'\n')
write('src/cot_navigation/training/atomic_checkpoint.py','"""Atomic checkpoint persistence shared by study trainers."""\n'+extract(adj,['save_atomic'])+'\n')
replace_definitions(adj,{'Samples','evaluate_adjacency','save_atomic'},
    'from cot_navigation.data.adjacency import Samples\nfrom cot_navigation.evaluation.adjacency import evaluate_adjacency\nfrom cot_navigation.training.atomic_checkpoint import save_atomic')
v5=studies/mapping['fixed_navigation_v5_20261026']/'train.py'
write('src/cot_navigation/evaluation/panels.py','"""Deterministic stratified panel selection."""\nimport random\n'+extract(v5,['panel_subset'])+'\n')
replace_definitions(v5,{'panel_subset'},'from cot_navigation.evaluation.panels import panel_subset')

# The trace validator is another core -> historical-script dependency.
audit=studies/mapping['scale_trace_audit_20261019']/'legacy_trace_audit.py'
write('src/cot_navigation/protocols/trace_audit.py',audit.read_text())
p=ROOT/'src/cot_navigation/analysis/legal_history.py'
s=p.read_text()
s=re.sub(r"    import sys\n    sys.path.insert\(0,str\(Path.cwd\(\)/'studies/scale_trace_audit'\)\)\n    from legacy_trace_audit import validate_trace",'    from cot_navigation.protocols.trace_audit import validate_trace',s)
p.write_text(s)

# A direct, explicit backend owns resource roots and reproducibility hashing.
write('src/cot_navigation/experiments/packed_backend.py','''"""Resource configuration for the shared packed trainer; no script imports."""
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
''')
p=ROOT/'src/cot_navigation/training/packed.py'
s=p.read_text()
s=s.replace('def main(argv=None, *, backend):\n    HERE = backend.here\n    load_v5, source_hash = backend.load_v5, backend.source_hash','def main(argv=None, *, backend=None):')
s=s.replace("    parser.add_argument('--repo-root', type=Path, required=True)","    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[3])\n    parser.add_argument('--backend', choices=['v6', 'v7'], default='v7')\n    parser.add_argument('--resources', type=Path)\n    parser.add_argument('--world-root', type=Path)\n    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')")
s=s.replace('choices=backend.protocols','choices=[\'direct\',\'path\',\'dfs_state\',\'dfs_nodes\']')
s=s.replace('    os.environ[\'CUDA_VISIBLE_DEVICES\'] = str(args.gpu)', '''    from cot_navigation.experiments.packed_backend import load_backend
    backend = backend or load_backend(args.backend, args.repo_root, resources=args.resources, world_root=args.world_root)
    HERE = backend.here
    if args.protocol not in backend.protocols:
        parser.error(f'{args.protocol} is not supported by this backend')
    source_hash = backend.source_hash
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)''')
s=s.replace('    v5 = load_v5(args.repo_root)','    from cot_navigation.evaluation.panels import panel_subset\n    from cot_navigation.evaluation.adjacency import evaluate_adjacency\n    from cot_navigation.data.adjacency import Samples')
s=s.replace("torch.device('cuda' if torch.cuda.is_available() else 'cpu')","torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)")
s=s.replace('v5.panel_subset','panel_subset')
s=s.replace("                legacy = v5.runtime(args.repo_root)\n                adj = legacy.evaluate_adjacency(raw, legacy.Samples(bank.world, 'direct'), device)","                adj = evaluate_adjacency(raw, Samples(bank.world, 'direct'), device)")
s += "\nif __name__ == '__main__':\n    main()\n"
p.write_text(s)
for name in ('v6','v7'):
    folder='fixed_navigation_protocols_v6' if name=='v6' else 'fixed_navigation_hard_v7'
    write(f'studies/{folder}/train_protocol.py',f'''"""Study-compatible entry point for the shared training engine."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cot_navigation.training.packed import main as _main, np_rows
from cot_navigation.experiments.packed_backend import load_backend

def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    return _main(argv, backend=load_backend('{name}', root))

if __name__ == '__main__':
    main()
''')

# Reference tests now use only fixtures/resources inside the independent tree.
p=ROOT/'tests/test_unified_framework.py';s=p.read_text()
s=s.replace("source=Path(SOURCE)/'studies'/version/'packed_bank.py'","source=ROOT/'tests/fixtures/packed_reference.py.txt'")
p.write_text(s)

# Preserve exact snapshots and distinguish migrated code from original archive.
(OUT/'migrated_text_files.json').write_text(json.dumps(changed,indent=2))
print(json.dumps({'studies':len(mapping),'relocated_files':len(changed)}))
