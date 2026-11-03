"""Native greedy decoding with per-row semantic early exit.

All normal rows use the same KV-cache operations and STOP handling as the
project generator. An early-exited row is marked censored, never assigned an
invented final answer. Illegal prefixes cannot become fully legal later.
"""
import torch
from cot_navigation.evaluation.rollout import pad_rows,_autocast,_gather_row,_decode_step


@torch.inference_mode()
def generate_checked(model,prompts,tok,precision,replays,margin):
    device=next(model.parameters()).device
    dtype=torch.bfloat16 if precision=='bf16' else torch.float32
    tokens,lengths=pad_rows(prompts,tok.pad,device)
    cache=model.new_cache(len(prompts),device,dtype)
    positions=torch.arange(tokens.shape[1],device=device)[None,:].expand(len(prompts),-1)
    with _autocast(device,precision):logits=model(tokens,positions=positions,cache=cache,cache_mode='prefill')
    cache.lengths=lengths.clone();next_logits=_gather_row(logits,lengths-1)
    active=[True]*len(prompts);body=[[] for _ in prompts];pred=[None]*len(prompts)
    interrupted=[False]*len(prompts);forced=[False]*len(prompts);censored=[False]*len(prompts)
    pending=[False]*len(prompts);counts=[0]*len(prompts);last_record=[0]*len(prompts)
    while any(active):
        picks=next_logits.argmax(-1).tolist()
        binary=next_logits[:,[tok.no,tok.yes]].argmax(-1).tolist() if any(pending) else None
        feed=[tok.pad]*len(prompts)
        for i in range(len(prompts)):
            if not active[i]:continue
            token=picks[i]
            if pending[i]:token=(tok.no,tok.yes)[binary[i]]
            elif margin and len(prompts[i])+counts[i]>=model.max_seq_len-margin-3 and token not in (tok.yes,tok.no):
                keep=last_record[i];body[i]=body[i][:keep];counts[i]=keep
                cache.lengths[i]=len(prompts[i])+keep;token=tok.id('<STOP>');forced[i]=True
            counts[i]+=1
            if token in (tok.yes,tok.no):pred[i]=int(token==tok.yes);active[i]=False
            elif token==tok.id('<STOP>'):
                interrupted[i]=True;pending[i]=True;feed[i]=token
            elif token==tok.id('<EOS>'):active[i]=False
            else:
                body[i].append(token)
                if token==tok.id(';'):last_record[i]=len(body[i])
                if replays[i].check(body[i]):active[i]=False;censored[i]=True
                elif len(prompts[i])+counts[i]>=model.max_seq_len:active[i]=False
                else:feed[i]=token
        if any(active):
            mask=torch.tensor(active,device=device,dtype=torch.bool)
            next_logits=_decode_step(model,cache,torch.tensor(feed,device=device)[:,None],precision,mask)
    return [dict(trace=body[i],pred=pred[i],interrupted=interrupted[i],forced_stop=forced[i],
        early_censored=censored[i],audit=replays[i].finish(body[i],pred[i],interrupted[i])) for i in range(len(prompts))]
