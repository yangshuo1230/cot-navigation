"""Landlock read isolation for the old repository and archived research.

The kernel policy is inherited by subprocesses and cannot be relaxed. This
does not rename, chmod or mount over any original project directory.
"""
import ctypes
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[2]
blocked=[root.with_name('cot-navigation'),root/'research',root/'.code_backups']
libc=ctypes.CDLL(None,use_errno=True)
READ=1|4
probes = [p for p in (
    root.with_name('cot-navigation')/'src/cot_navigation/config.py',
    root/'research/fixed_navigation_v5_20261026/train.py',
    *list((root/'.code_backups').glob('*.zip'))[:1],
) if p.is_file()]


class Ruleset(ctypes.Structure):
    _fields_=[('handled_access_fs',ctypes.c_uint64)]

class Beneath(ctypes.Structure):
    _pack_=1
    _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]

def check(value,what):
    if value<0:
        error=ctypes.get_errno()
        raise OSError(error,what+': '+os.strerror(error))
    return value

abi=check(libc.syscall(444,0,0,1),'Landlock ABI query')
attr=Ruleset(READ)
fd=check(libc.syscall(444,ctypes.byref(attr),ctypes.sizeof(attr),0),'create ruleset')

def allow(path):
    handle=os.open(path,os.O_PATH|os.O_CLOEXEC)
    try:
        access=READ if path.is_dir() else 1|4
        rule=Beneath(access,handle)
        check(libc.syscall(445,fd,1,ctypes.byref(rule),0),'allow '+str(path))
    finally:os.close(handle)

def traverse(parent):
    for child in parent.iterdir():
        resolved=child.resolve()
        if any(resolved==p or p in resolved.parents for p in blocked):continue
        if any(resolved in p.parents for p in blocked):
            traverse(child)
        else:
            try:allow(child)
            except FileNotFoundError:pass

traverse(Path('/'))
check(libc.prctl(38,1,0,0,0),'no_new_privs')
check(libc.syscall(446,fd,0),'restrict_self')
os.close(fd)
for path in probes:
    try:
        path.read_bytes()
    except PermissionError:continue
    raise RuntimeError('Isolation did not deny '+str(path))
print('LANDLOCK ABI',abi,'DENIED',*[str(p) for p in blocked],flush=True)
os.chdir(root)
os.environ.update(PYTHONPATH=str(root/'src')+':'+str(root),PYTHONDONTWRITEBYTECODE='1',
    PYTHONPYCACHEPREFIX=str(root/'migration/standalone/pycache'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
args=sys.argv[1:] or [str(root/'.venv/bin/python'),'-m','pytest','tests','-o','addopts=','-q']
os.execvp(args[0],args)
