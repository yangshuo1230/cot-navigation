from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
output=ROOT/'migration/standalone/research_tests';output.mkdir(exist_ok=True)
unit=['curriculum_loss_swa/test_record_swa.py','activation_patching/test_patch.py',
      'activation_patching_expanded/test_engine.py','activation_patching_expanded/test_data.py']
scripts=['fixed100_layer_probes/test_methods.py','fixed100_legality/test_replay.py','fixed100_legality/test_stream.py']
reports=[]
for name in unit+scripts:
    source=ROOT/'studies'/name
    if name in unit:
        argv=[sys.executable,'-m','pytest',str(source),'-o','addopts=','-q']
    else:
        # Redirect only the test receipt, preserving existing study results.
        content=source.read_text()
        for receipt in ('stream_fidelity.json','validator_tests.json'):
            content=content.replace("(OUT/'"+receipt+"')",repr(str(output/receipt)))
            content=content.replace(repr(str(output/receipt))+'.write_text', 'Path('+repr(str(output/receipt))+').write_text')
        copy=output/(source.parent.name+'__'+source.name);copy.write_text(content)
        argv=[sys.executable,str(copy)]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'2','TORCHINDUCTOR_COMPILE_THREADS':'1',
         'PYTHONPATH':str(source.parent)+':'+str(ROOT/'src')+':'+str(ROOT), 'OMP_NUM_THREADS':'2'}
    try:
        result=subprocess.run(argv,cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
        text=result.stdout+result.stderr;code=result.returncode
    except subprocess.TimeoutExpired as exc:
        text=str(exc);code=124
    (output/(name.replace('/','__')+'.log')).write_text(text)
    reports.append({'test':name,'returncode':code,'tail':text[-2400:]})
(output/'summary.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2))
