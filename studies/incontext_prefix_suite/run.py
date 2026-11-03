from pathlib import Path
import argparse,os,sys,json,fcntl
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--gpu',required=True,type=int);p.add_argument('--resume',action='store_true');args=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
    sys.path.insert(0,str(HERE/'src'))
    import torch
    torch.set_num_threads(2)
    from cot_navigation.config import load_config
    from runtime import PrefixTrainer
    manifest=json.loads((HERE/'manifest.json').read_text());job=next(j for j in manifest['jobs'] if j['id']==args.job)
    cfg=load_config([job['config']],['runtime.cuda_device=0'])
    out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'COMPLETE.json').exists():return
    (out/'experiment_job.json').write_text(json.dumps(job,indent=2))
    checkpoint=out/'checkpoints/latest.pt'
    if checkpoint.exists() and not args.resume:raise RuntimeError('Refusing to overwrite existing checkpoint')
    t=PrefixTrainer(cfg,job,resume=str(checkpoint) if args.resume else None)
    try:
        t.fit();assert (out/'final.json').is_file()
        (out/'COMPLETE.json').write_text(json.dumps(dict(step=t.completed,queries_seen=t.queries_seen)))
    except KeyboardInterrupt:
        t._save();t.tracker.finish();raise
if __name__=='__main__':main()
