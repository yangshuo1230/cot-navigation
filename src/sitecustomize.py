"""Protect the source project from writes by migrated Python processes."""
import os
from pathlib import Path
import sys

PROTECTED = Path(os.environ.get('COT_NAV_PROTECTED_ROOT', '/mnt/ningxuefei/yangshuo/cot-navigation')).resolve()

def _check(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(value)).resolve()
    if path == PROTECTED or PROTECTED in path.parents:
        raise PermissionError(f'Original project is read-only in this environment: {path}')

def _audit(event, args):
    if event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            _check(args[0])
    elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.chown','os.utime','os.truncate'}:
        _check(args[0])
    elif event in {'os.rename','os.link'}:
        _check(args[0]); _check(args[1])
    elif event == 'os.symlink':
        _check(args[1])

sys.addaudithook(_audit)

if os.environ.get('COT_NAV_HELP_CHECK') == '1':
    _project = Path(__file__).resolve().parent.parent
    def _help_check(event, args):
        if event in {'subprocess.Popen', 'os.system'}:
            raise PermissionError('Help audit forbids launching child processes')
        targets = []
        if event == 'open':
            mode, flags = args[1], args[2]
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
                targets = [args[0]]
        elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.chown','os.utime','os.truncate'}:
            targets = [args[0]]
        elif event in {'os.rename','os.link'}:
            targets = list(args[:2])
        for value in targets:
            if isinstance(value, (str,bytes,os.PathLike)):
                path = Path(os.fsdecode(value)).resolve()
                if path == _project or _project in path.parents:
                    raise PermissionError(f'Help audit forbids project writes: {path}')
    sys.addaudithook(_help_check)
