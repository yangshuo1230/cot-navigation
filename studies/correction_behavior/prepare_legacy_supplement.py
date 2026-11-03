"""Audit coverage and reference existing valid legacy trajectories without copying."""
import gzip
import importlib.util
import json
from pathlib import Path

P=Path(__file__).resolve().parent;ROOT=P.parents[1]


def main():
    spec=importlib.util.spec_from_file_location('legacy_behavior_prepare',P/'run.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    manifest=json.loads((P/'cohorts/legacy6_s0123_supplement_20261102.json').read_text())
    out,plan=base.prepare(manifest);refs=ROOT/manifest['reference_results_dir'];refs.mkdir(exist_ok=True,parents=True)
    old=ROOT/'studies/correction_reproduction/error_continuation';coverage=[]
    for model in plan['models']:
        for task in manifest['tasks']:
            name=f"{model['training_arm']}_s{model['seed']}_{task}.json.gz"
            original=old/'results'/name;detailed=old/'dependencies/results'/name
            with gzip.open(original,'rt') as f:raw=json.load(f)
            assert raw['complete'] and len(raw['rows'])==256
            assert [r['query'] for r in raw['rows']]==plan['tasks'][task]['queries']
            path=original
            if detailed.exists():
                with gzip.open(detailed,'rt') as f:d=json.load(f)
                assert d['complete'] and [(r['query'],r['trace_sha256']) for r in d['rows']]==[(r['query'],r['trace_sha256']) for r in raw['rows']]
                path=detailed
            dest=refs/f"{model['id']}__{task}.json.gz"
            if dest.exists():assert dest.resolve()==path.resolve()
            else:dest.symlink_to(path.resolve())
            coverage.append(dict(model=model['id'],task=task,natural_complete=True,dependencies_complete=detailed.exists(),source=str(path),source_sha256=base.sha_file(path)))
    base.atomic_json(out/'existing_coverage.json',coverage)
    report=['# 旧版6层0%/30%、四seed补缺计划','','仅使用dfs_correction_repro_20261028的8模型，排除已废弃depth-rate批次。',
        '已完成：40/40自然任务、40/40长度/错误历史统计；20/40节点级细查（seed0/1）。',
        '补做：seed2/3节点级细查、统一MISTAKE检出/漏标/修复/类型偏好、错误CoT答对细分、40/40 oracle配对任务。',
        '原有探针/attention等不属于本次缺项，不重跑。旧结果保留且通过符号引用复用，不复制raw数据。',
        'oracle配对必需的自然基线同时捕获所有缺失原始计数和结构关系，不另起自然生成。',
        'matched_clean配置含0.3抽样率，但reproduction_identity证实使用matched_clean采样器；实际训练错误率0%。',
        '新自然基线逐条对照旧轨迹hash；若不一致，不把两者的token级关系混合。','']
    (out/'SUPPLEMENT_PLAN.md').write_text('\n'.join(report),encoding='utf-8')
    print('EXISTING: natural',len(coverage),'structural',sum(x['dependencies_complete'] for x in coverage),'models',len(plan['models']))


if __name__=='__main__':main()
