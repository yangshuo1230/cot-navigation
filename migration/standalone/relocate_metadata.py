from pathlib import Path
import json
import re
import shutil

root=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
out=root/'migration/standalone'
mapping=json.loads((out/'study_paths.json').read_text())
(root/'configs/study_paths.json').write_text(json.dumps(mapping,indent=2))

def remap(text):
    text=text.replace('/mnt/ningxuefei/yangshuo/cot-navigation/',str(root)+'/')
    # Restrict experiment identifier renaming to path components.
    text=text.replace('/research/','/studies/').replace('research/','studies/')
    for old,new in mapping.items():
        text=text.replace('studies/'+old+'/', 'studies/'+new+'/')
        text=text.replace('studies/'+old+'"', 'studies/'+new+'"')
        text=text.replace("studies/"+old+"'", "studies/"+new+"'")
    return text

changed=[]
for folder in ('studies','configs'):
    for p in (root/folder).rglob('*'):
        if not p.is_file() or p.suffix not in {'.json','.yaml','.yml','.sh'}:continue
        if p.name=='study_paths.json':continue
        try:original=p.read_text()
        except UnicodeError:continue
        updated=remap(original)
        if original==updated:continue
        rel=p.relative_to(root)
        backup=out/'metadata_originals'/rel
        backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,backup)
        p.write_text(updated);changed.append(str(rel))
(out/'relocated_metadata.json').write_text(json.dumps(changed,indent=2))

p=root/'src/cot_navigation/runtime/workspace.py'
s=p.read_text().replace('import re','import re\nimport json\nfrom functools import lru_cache')
s += '''
@lru_cache(maxsize=1)
def study_paths():
    return json.loads((ROOT / 'configs/study_paths.json').read_text())

def artifact_path(value):
    value = relocate(str(value))
    value = value.replace('/research/', '/studies/')
    if value.startswith('research/'):
        value = 'studies/' + value[len('research/'):]
    for old, new in study_paths().items():
        value = value.replace('studies/'+old+'/', 'studies/'+new+'/')
    return Path(value)
'''
p.write_text(s)
p=root/'src/cot_navigation/config.py'
s=p.read_text().replace('        source = _load_yaml_tree(Path(path))','        from cot_navigation.runtime.workspace import artifact_path\n        source = _load_yaml_tree(artifact_path(path))')
p.write_text(s)
print(json.dumps({'metadata_files_relocated':len(changed)}))
