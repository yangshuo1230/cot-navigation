from pathlib import Path
import ast
import json
import re

root=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
p=root/'src/cot_navigation/training/packed.py'
s=p.read_text().replace("    parser.add_argument('--resume', action='store_true')","    parser.add_argument('--resume', action='store_true')\n    parser.add_argument('--stop-after', type=int, help='Checkpoint and stop after this absolute update; useful for controlled interruption')")
s=s.replace('    args.repo_root = args.repo_root.resolve()',"    args.repo_root = args.repo_root.resolve()\n    if args.stop_after is not None and args.stop_after < 1:\n        parser.error('--stop-after must be positive')")
s=s.replace("            if step == 20000:\n                save('step020000.pt')","            if step == 20000:\n                save('step020000.pt')\n            if args.stop_after is not None and step >= args.stop_after and step < cfg['task_steps']:\n                save('latest.pt')\n                return")
p.write_text(s)

# Before executing help checks, enumerate imports and source paths for review.
report={'historical_path_literals':[], 'external_absolute_paths':[], 'sources':0}
for folder in ('src','scripts','studies','tests'):
    for p in (root/folder).rglob('*.py'):
        if '__pycache__' in p.parts:continue
        tree=ast.parse(p.read_text());report['sources']+=1
        for n in ast.walk(tree):
            if not isinstance(n,ast.Constant) or not isinstance(n.value,str):continue
            value=n.value
            if ('research/' in value or '/research' in value) and '\n' not in value:
                report['historical_path_literals'].append({'file':str(p.relative_to(root)),'line':n.lineno,'value':value})
            if ('/mnt/' in value or '/home/' in value) and len(value)<300:
                report['external_absolute_paths'].append({'file':str(p.relative_to(root)),'line':n.lineno,'value':value})
(root/'migration/standalone/dependency_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:len(v) if isinstance(v,list) else v for k,v in report.items()}))
