import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch
import yaml

from cot_navigation.experiments.workflow import plan
from cot_navigation.runtime.workspace import relocate, ROOT, SOURCE
from cot_navigation.data.packed_bank import PackedBank
from cot_navigation.training.packed_loss import logical_loss


def test_plan_matrix_is_read_only_and_deterministic(tmp_path):
    config = ROOT/'configs/workflows/context_protocols.yaml'
    first = plan(config, run_name='unit_matrix')
    assert first == plan(config, run_name='unit_matrix')
    assert len(first[1]) == 6
    assert len({s.name for s in first[1]}) == 6
    assert not (ROOT/'outputs/workflows/unit_matrix').exists()
    assert all('cot_navigation.cli' in s.argv for s in first[1])


@pytest.mark.parametrize('steps', [
    [{'name':'a','kind':'studies','script':'../cot-navigation/scripts/scale.py'}],
    [{'name':'a','kind':'studies','script':'scripts/scale.py','depends_on':['b']}],
    [{'name':'a','kind':'studies','script':'scripts/scale.py','matrix':{'x':[]}}],
    [{'name':'a','kind':'studies','script':'scripts/scale.py','matrix':{'x':[1,2]}}],
])
def test_invalid_plans_fail_before_writes(tmp_path, steps):
    path=tmp_path/'plan.yaml'
    path.write_text(yaml.safe_dump({'version':1,'run_name':'invalid','steps':steps}))
    with pytest.raises(ValueError):
        plan(path)
    assert not (ROOT/'outputs/workflows/invalid').exists()


def test_relocation_keeps_artifacts_and_values_unchanged():
    original={'data':{'root':SOURCE+'/data/a'},'sets':['x='+SOURCE+'/data/b'],'weight':1.0}
    result=relocate(original)
    assert result['data']['root']==str(ROOT/'data/a')
    assert result['sets']==['x='+str(ROOT/'data/b')]
    assert original['data']['root'].startswith(SOURCE+'/')


def test_write_barrier_checks_before_os_operation():
    result=subprocess.run([sys.executable,'-c',
        "import sys; sys.audit('open', '"+SOURCE+"/MUST_NOT_EXIST', 'w', 0)"],capture_output=True,text=True)
    assert result.returncode != 0 and 'Original project is read-only' in result.stderr
    # Audit rejection happens before any OS file operation.


@pytest.mark.parametrize('version', ['fixed_navigation_protocols_v6','fixed_navigation_hard_v7'])
@pytest.mark.parametrize('protocol', ['direct','dfs_state','dfs_nodes'])
def test_packed_data_loss_and_gradient_match_original(version, protocol):
    # Original source is read as text, but roots point exclusively at the copy.
    source=ROOT/'tests/fixtures/packed_reference.py.txt'
    here=ROOT/'studies'/version
    base=here/'worlds' if 'hard_v7' in version else ROOT/'studies/fixed_navigation_v5'
    from cot_navigation.experiments.artifacts import sha_file,digest
    namespace={'HERE':here,'BASE':base,'sha_file':sha_file,'digest':digest}
    code=source.read_text().replace('from prepare import HERE, BASE, sha_file, digest','')
    exec(compile(code, str(source), 'exec'),namespace)
    manifest=json.loads((here/'manifest.json').read_text())
    cfg=json.loads((here/'experiment.json').read_text())
    meta=manifest['worlds'][0]
    old=namespace['PackedBank'](meta,protocol,torch.device('cpu'))
    new=PackedBank(meta,protocol,torch.device('cpu'),bank_root=here,world_root=base)
    ids=old.sample(random.Random(1901),random.Random(1902),cfg)
    assert ids==new.sample(random.Random(1901),random.Random(1902),cfg)
    assert old.microbatches(ids,cfg)==new.microbatches(ids,cfg)
    for a,b in zip(old.batch(ids[:3]),new.batch(ids[:3])):
        assert torch.equal(a,b)
    # Keep gradient equivalence inexpensive even for full DFS traces.
    tokens=torch.tensor([[1,2,3,4,5,6,7,0],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,0,0]])
    lengths=torch.tensor([7,8,6]); answers=torch.tensor([5,6,0]); groups=torch.tensor([0,1,2])
    denom=torch.tensor([2.,3.,1.,1.])
    logits=torch.randn(3,8,16,requires_grad=True)
    clone=logits.detach().clone().requires_grad_()
    a=namespace['logical_loss'](logits,tokens,lengths,answers,groups,denom,1.,1.,protocol!='direct')
    b=logical_loss(clone,tokens,lengths,answers,groups,denom,1.,1.,protocol!='direct')
    assert torch.equal(a,b)
    a[0].backward(); b[0].backward()
    assert torch.equal(logits.grad,clone.grad)


def test_microbatch_global_objective_matches_full_batch():
    tokens=torch.randint(0,12,(4,9)); lengths=torch.tensor([9,8,7,6])
    answers=torch.tensor([7,6,5,0]); groups=torch.tensor([0,1,1,2])
    denominator=torch.tensor([3.,3.,1.,2.])
    full=torch.randn(4,9,12,requires_grad=True)
    split=full.detach().clone().requires_grad_()
    a=logical_loss(full,tokens,lengths,answers,groups,denominator,2.,.5,True)
    b=sum(logical_loss(split[s],tokens[s],lengths[s],answers[s],groups[s],denominator,2.,.5,True)
          for s in [slice(0,1),slice(1,3),slice(3,4)])
    torch.testing.assert_close(a,b)
    a[0].backward(); b[0].backward()
    torch.testing.assert_close(full.grad,split.grad)
