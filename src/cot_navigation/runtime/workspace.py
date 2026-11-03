"""Project-relative resolution of imported artifact paths."""
from pathlib import Path
import re
import json
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[3]
SOURCE = '/mnt/ningxuefei/yangshuo/cot-navigation'

def relocate(value):
    if isinstance(value, dict):
        return {k: relocate(v) for k,v in value.items()}
    if isinstance(value, list):
        return [relocate(v) for v in value]
    if isinstance(value, tuple):
        return tuple(relocate(v) for v in value)
    if isinstance(value, str):
        for prefix in (SOURCE+'-unified/', SOURCE+'/'):
            value = value.replace(prefix, str(ROOT)+'/')
    return value

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
