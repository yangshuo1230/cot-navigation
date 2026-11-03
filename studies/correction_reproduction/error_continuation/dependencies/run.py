"""Reuse natural generation; retain only the additional structural annotations."""
import importlib.util
import json
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('continuation_generation',P.parent/'run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from cot_navigation.analysis.error_dependencies import dependencies
from cot_navigation.experiments.artifacts import atomic_json,sha_file

original=base.audit_continuation


def detailed(trace,adj,src,dst,tok):
    result=original(trace,adj,src,dst,tok)
    result['dependencies']=dependencies(trace,adj,src,dst,tok)
    return result


if __name__=='__main__':
    atomic_json(P/'mechanism_plan.json',dict(gpu=0,initial_model_seeds=[0,1],queries_per_task=256,
        natural_generation=True,graph_tasks=base.TASKS,
        provenance={str(f.relative_to(base.ROOT)):sha_file(f) for f in [Path(__file__),base.ROOT/'src/cot_navigation/analysis/error_dependencies.py']},
        note='Additional natural replay is necessary because old summaries omit node identities. No token intervention.'))
    base.P=P;base.audit_continuation=detailed
    if '--seeds' not in sys.argv:sys.argv+=['--seeds','0','1']
    base.main()
