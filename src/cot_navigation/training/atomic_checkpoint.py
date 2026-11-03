"""Atomic checkpoint persistence shared by study trainers."""
def save_atomic(path,payload):
    import torch
    temp=path.with_suffix('.writing')
    torch.save(payload,temp);temp.replace(path)

