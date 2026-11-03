"""Autoregressive full adjacency-list evaluation."""
from collections import Counter
def evaluate_adjacency(raw,samples,device):
    """Autoregressive complete-list recovery: no true-degree oracle and no top-k."""
    from cot_navigation.evaluation.rollout import generate
    tok=samples.tok;raw.eval()
    rows=[samples.knowledge(('adj',u)) for u in range(len(samples.nodes))]
    counts=Counter();predictions=[]
    for start in range(0,len(rows),128):
        part=rows[start:start+128]
        outputs=generate(raw,[r.tokens[:r.prompt_len] for r in part],'path',tok,
            'bf16' if device.type=='cuda' else 'fp32',max_new_tokens=32)
        for u,row,out in zip(range(start,start+len(part)),part,outputs):
            generated=out['generated'];expected=row.tokens[row.prompt_len:]
            ended=bool(generated) and generated[-1]==tok.id('<EOS>')
            body=generated[:-1] if ended else generated
            is_empty=body==[tok.id('<EXHAUST>')]
            legal=ended and (is_empty or (bool(body) and len(body)==len(set(body)) and
                                           all(x in samples.nodes for x in body)))
            true={samples.nodes[v] for v in samples.world['adj'][u]}
            predicted=set() if is_empty else set(body)
            exact=legal and predicted==true
            counts['nodes']+=1;counts['exact_set']+=exact;counts['exact_sequence']+=generated==expected
            counts['true_edges']+=len(true);counts['predicted_edges']+=len(predicted) if legal else 0
            counts['correct_edges']+=len(predicted&true) if legal else 0
            counts['valid_format']+=legal
            if not true:counts['empty_sources']+=1;counts['empty_correct']+=exact
            predictions.append(dict(source=u,expected=expected,generated=generated,exact=bool(exact)))
    raw.train()
    return dict(n_sources=counts['nodes'],exact_set_accuracy=counts['exact_set']/counts['nodes'],
        exact_sequence_accuracy=counts['exact_sequence']/counts['nodes'],
        edge_recall=counts['correct_edges']/max(1,counts['true_edges']),
        edge_precision=counts['correct_edges']/max(1,counts['predicted_edges']),
        empty_set_accuracy=counts['empty_correct']/max(1,counts['empty_sources']),
        empty_sources=counts['empty_sources'],format_valid_rate=counts['valid_format']/counts['nodes'],
        predictions=predictions)

