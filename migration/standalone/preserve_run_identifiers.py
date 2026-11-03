"""Only relocate study directory names, never embedded run/file identifiers."""
from pathlib import Path
import io
import json
import re
import tokenize

ROOT=Path(__file__).resolve().parents[2]
mapping=json.loads((ROOT/'configs/study_paths.json').read_text())
changes=[]
for source in (ROOT/'research').rglob('*.py'):
    if '__pycache__' in source.parts:continue
    rel=source.relative_to(ROOT/'research');parts=rel.parts
    target=ROOT/'studies'/mapping.get(parts[0],parts[0])
    for part in parts[1:]:target/=part
    if not target.is_file():continue
    original=source.read_text();current=target.read_text()
    restore={}
    for token in tokenize.generate_tokens(io.StringIO(original).readline):
        if token.type!=tokenize.STRING:continue
        for old,new in mapping.items():
            for match in re.finditer(re.escape(old)+r'([A-Za-z0-9_.-]+)',token.string):
                suffix=match.group(1)
                restore[new+suffix]=old+suffix
    if not restore:continue
    tokens=[];used=[]
    for token in tokenize.generate_tokens(io.StringIO(current).readline):
        if token.type==tokenize.STRING:
            text=token.string
            for before,after in sorted(restore.items(),key=lambda item:-len(item[0])):
                if before in text:
                    text=text.replace(before,after);used.append({'before':before,'after':after})
            token=token._replace(string=text)
        tokens.append(token)
    if used:
        target.write_text(tokenize.untokenize(tokens))
        changes.append({'file':str(target.relative_to(ROOT)),'identifiers':used})
(ROOT/'migration/standalone/preserved_run_identifiers.json').write_text(json.dumps(changes,indent=2))
print(json.dumps({'files':len(changes),'restored_identifiers':sum(len(r['identifiers']) for r in changes)}))
