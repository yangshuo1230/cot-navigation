"""Incremental grounded state replay; stop at the first proven semantic error.

Frontier order is exact. VIS and CANDS are semantically sets; chronological
VIS order and PUSH-first-CANDS are separately reported protocol constraints.
No fixed reference traversal order is imposed.
"""
from collections import Counter


class StateReplay:
    def __init__(self,adj,source,target,tok,fifo=False):
        self.adj=adj;self.target=target;self.tok=tok;self.fifo=fifo
        self.frontier=[source];self.history=[source];self.seen={source}
        self.start=0;self.records=0;self.error=None;self.terminal=None
        self.protocol_errors=Counter();self.last_record=[];self.previous_record=[]
        self.action_names={tok.id('<'+x+'>'):x for x in ('PUSH','POP','FOUND','EXHAUST')}

    def fail(self,kind,detail=None):
        self.error={'kind':kind,'record':self.records,'detail':detail or {}}
        return True

    def check(self,body):
        last=body[-1]
        if last!=self.tok.id(';'):
            if len(body)-self.start>3*len(self.adj)+6:
                return self.fail('overlong_record')
            if last not in self.adj and last not in self.action_names and last!=self.tok.id('|'):
                return self.fail('non_protocol_token',{'token':last})
            return False
        raw=body[self.start:-1];self.start=len(body)
        self.previous_record=self.last_record;self.last_record=list(raw)
        if self.terminal:return self.fail('record_after_terminal')
        fields=[[]]
        for token in raw:
            if token==self.tok.id('|'):fields.append([])
            else:fields[-1].append(token)
        if len(fields)!=4:return self.fail('malformed_fields')
        ff,vv,cc,aa=fields
        if ff!=self.frontier:return self.fail('frontier_mismatch',{'expected':self.frontier.copy(),'shown':ff})
        if len(vv)!=len(set(vv)) or set(vv)!=self.seen:
            return self.fail('visited_mismatch',{'expected':self.history.copy(),'shown':vv})
        if vv!=self.history:self.protocol_errors['visited_order']+=1
        current=self.frontier[0 if self.fifo else -1]
        eligible=set(self.adj[current])-self.seen
        if set(cc)!=eligible or len(cc)!=len(set(cc)):
            return self.fail('candidate_mismatch',{'current':current,'visited':self.history.copy(),
                'expected':sorted(eligible),'shown':cc,'missing':sorted(eligible-set(cc)),
                'extra':sorted(set(cc)-eligible),'shown_action':aa})
        expected='FOUND' if current==self.target else 'PUSH' if eligible else 'POP' if len(self.frontier)>1 else 'EXHAUST'
        action=self.action_names.get(aa[0]) if aa else None
        if action!=expected or len(aa)!=(2 if expected=='PUSH' else 1):
            return self.fail('wrong_action',{'current':current,'expected':expected,'shown':aa})
        if action=='PUSH':
            if aa[1] not in eligible:return self.fail('illegal_push',{'current':current,'chosen':aa[1]})
            if aa[1]!=cc[0]:self.protocol_errors['push_not_first_candidate']+=1
            self.frontier.append(aa[1]);self.seen.add(aa[1]);self.history.append(aa[1])
        elif action=='POP':self.frontier.pop(0 if self.fifo else -1)
        else:self.terminal=action
        self.records+=1
        return False

    def finish(self,body,pred,interrupted=False):
        if not self.error:
            if self.start!=len(body):self.fail('unfinished_record')
            elif not self.terminal:self.fail('legal_prefix_stopped' if interrupted else 'missing_terminal')
            elif pred!=int(self.terminal=='FOUND'):self.fail('terminal_answer_mismatch',{'terminal':self.terminal,'pred':pred})
        valid=self.error is None and not interrupted
        return dict(semantic_valid=valid,protocol_valid=valid and not self.protocol_errors,
            prefix_valid_records=self.records,first_error=self.error,
            protocol_errors=dict(self.protocol_errors),terminal=self.terminal,
            last_record=self.last_record,previous_record=self.previous_record,
            censored_after_first_error=bool(self.error))


class ActionReplay:
    """Action-only replay plus full-state observer on a preselected panel.

    Display errors do not stop decoding here. Actions use the actual graph
    and independently maintained frontier/visited, not the printed state.
    """
    def __init__(self,adj,source,target,tok,fifo=False):
        self.adj=adj;self.frontier=[source];self.seen={source};self.target=target
        self.tok=tok;self.fifo=fifo;self.start=0;self.records=0;self.error=None;self.terminal=None
        self.full=StateReplay(adj,source,target,tok,fifo)
        self.names=self.full.action_names;self.last_record=[];self.previous_record=[]

    def fail(self,kind,detail=None):
        self.error=dict(kind=kind,record=self.records,detail=detail or {});return True

    def check(self,body):
        if not self.full.error:self.full.check(body)
        if body[-1]!=self.tok.id(';'):
            if len(body)-self.start>3*len(self.adj)+6:return self.fail('overlong_record')
            return False
        raw=body[self.start:-1];self.start=len(body)
        self.previous_record=self.last_record;self.last_record=list(raw)
        if self.terminal:return self.fail('record_after_terminal')
        positions=[i for i,t in enumerate(raw) if t in self.names]
        if len(positions)!=1:return self.fail('malformed_action')
        aa=raw[positions[0]:];action=self.names[aa[0]]
        if len(aa)!=(2 if action=='PUSH' else 1):return self.fail('malformed_action')
        cur=self.frontier[0 if self.fifo else -1];eligible=set(self.adj[cur])-self.seen
        expected='FOUND' if cur==self.target else 'PUSH' if eligible else 'POP' if len(self.frontier)>1 else 'EXHAUST'
        if action!=expected:return self.fail('wrong_action',dict(current=cur,expected=expected,shown=action))
        if action=='PUSH':
            if aa[1] not in eligible:return self.fail('illegal_push',dict(current=cur,chosen=aa[1],eligible=sorted(eligible)))
            self.frontier.append(aa[1]);self.seen.add(aa[1])
        elif action=='POP':self.frontier.pop(0 if self.fifo else -1)
        else:self.terminal=action
        self.records+=1
        return False

    def finish(self,body,pred,interrupted=False):
        if not self.error:
            if self.start!=len(body):self.fail('unfinished_record')
            elif not self.terminal:self.fail('legal_prefix_stopped' if interrupted else 'missing_terminal')
            elif pred!=int(self.terminal=='FOUND'):self.fail('terminal_answer_mismatch')
        valid=self.error is None and not interrupted
        return dict(semantic_valid=valid,protocol_valid=valid,prefix_valid_records=self.records,
            first_error=self.error,protocol_errors={},terminal=self.terminal,
            last_record=self.last_record,previous_record=self.previous_record,
            full_state=self.full.finish(body,pred,interrupted))


class SequenceReplay:
    def __init__(self,adj,source,target,label,protocol,tok,validator):
        self.adj=adj;self.source=source;self.target=target;self.label=label
        self.protocol=protocol;self.tok=tok;self.validator=validator;self.error=None;self.result=None
        self.last_record=[];self.previous_record=[]

    def check(self,body):
        result=self.validator(body,self.adj,self.source,self.target,self.label,self.protocol)
        self.result=result
        if not result['prefix_valid']:
            self.error=result['first_error'];return True
        return False

    def finish(self,body,pred,interrupted=False):
        result=self.validator(body,self.adj,self.source,self.target,self.label,self.protocol)
        semantic=result['complete_valid'] and not interrupted
        return dict(semantic_valid=semantic,protocol_valid=semantic and pred==self.label,
            prefix_valid_records=result['valid_prefix_units'],first_error=self.error or result['first_error'],
            protocol_errors={},terminal=None,last_record=body[-100:],previous_record=[],
            local_path_prefix_legal=result['prefix_valid'])
