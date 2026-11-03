from pathlib import Path
root=Path('/mnt/ningxuefei/yangshuo/cot-navigation-unified')
p=root/'src/cot_navigation/evaluation/packed.py'
p.write_text(p.read_text().replace('    from prepare import digest\n',''))
p=root/'src/sitecustomize.py'
s=p.read_text()
s += '''
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
'''
p.write_text(s)
