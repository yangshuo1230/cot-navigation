"""Study-compatible entry point for the shared training engine."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cot_navigation.training.packed import main as _main, np_rows
from cot_navigation.experiments.packed_backend import load_backend

def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    return _main(argv, backend=load_backend('v6', root))

if __name__ == '__main__':
    main()
