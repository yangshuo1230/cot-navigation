"""CPU-only independent label and reference-format checks for prepared test data."""
import json
import sys
from pathlib import Path

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from run import ROOT, DATA, OUT, MODELS, read_gzip
from cot_navigation.config import load_config
from cot_navigation.data.graph import adj_from_presentation, structure_hash
from cot_navigation.data.reachability import bfs_reaches
from cot_navigation.protocols.render import context_prompt, render_task
from cot_navigation.protocols.tokenizer import Tokenizer
from cot_navigation.protocols.correction_replay import replay, self_test
from cot_navigation.experiments.scheduler import write_json
from cot_navigation.data.shards import file_sha256


def main():
    manifest=json.loads((DATA/'manifest.json').read_text())
    cfg=load_config([MODELS/'correction/s0/resolved_config.yaml'])
    tok=Tokenizer.from_config(cfg)
    self_test(tok)
    seen=set()
    labels=0
    references=0
    cells=[]
    for cell in manifest['cells']:
        assert file_sha256(DATA/cell['file'])==cell['sha256']
        rows=read_gzip(DATA/cell['file'])
        assert len(rows)==512 and sum(r['label'] for r in rows)==256
        for i,r in enumerate(rows):
            edges=list(zip(r['edge_src'],r['edge_dst']))
            adj=adj_from_presentation(cell['n'],edges)
            assert len(set(edges))==cell['m'] and all(0<=u<v<cell['n'] for u,v in edges)
            assert structure_hash(cell['layers'],cell['width'],edges)==r['struct_hash']
            assert r['struct_hash'] not in seen
            seen.add(r['struct_hash'])
            assert bfs_reaches(adj,r['s'],r['t'])==bool(r['label'])
            labels+=1
            if i%16==0:
                nodes=[tok.node_id(v) for v in r['node_token']]
                prompt=context_prompt(tok,nodes,edges,r['s'],r['t'])
                assert len(prompt)==3*cell['m']+4<4096
                ref=render_task(tok,'dfs_state',r['label'],prompt,adj,r['s'],r['t'],nodes)
                token_adj={nodes[u]:[nodes[v] for v in vs] for u,vs in enumerate(adj)}
                result=replay(ref.tokens[ref.prompt_len:ref.answer_index],r['label'],token_adj,
                              nodes[r['s']],nodes[r['t']],tok,prompt_len=len(prompt),label=r['label'])
                assert result['literal_valid'] and result['answer_and_trace_valid'], (cell['name'],i,result)
                references+=1
        cells.append(dict(name=cell['name'],n=len(rows),positive=256,negative=256))
    report=dict(passed=True,cpu_only=True,labels_checked_with_bfs=labels,
                unique_graphs=len(seen),full_reference_replay_checks=references,cells=cells,
                manifest_sha256=file_sha256(DATA/'manifest.json'))
    write_json(OUT/'verification.json',report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
