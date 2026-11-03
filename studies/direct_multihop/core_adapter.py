"""Reuse the verified pure-Direct math; do not edit the earlier study."""
from pathlib import Path
import importlib.util
PATH=Path(str(next(_p for _p in __import__('pathlib').Path(__file__).resolve().parents if (_p / 'pyproject.toml').is_file() and (_p / 'studies').is_dir())) + '/studies/direct_learning/train.py')
spec=importlib.util.spec_from_file_location('onehop_math',PATH)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
