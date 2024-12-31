import torch
from mash_diffusion.Model.mash_net import MashNet
from mash_diffusion.Method.model_size import getModelFLOPSAndParamsNum

if __name__ == "__main__":
    batch_size = 1
    context_num = 1397
    context_dim = 1024

    model = MashNet(
        n_latents=400,
        mask_degree=3,
        sh_degree=2,
        context_dim=context_dim,
        n_heads=8,
        d_head=64,
        depth=24)
    inputs = (torch.randn(batch_size, 400, 25), torch.randn(batch_size, context_num, context_dim), torch.randn(batch_size))

    flops, params = getModelFLOPSAndParamsNum(model, inputs)

    print('model_FLOPs:', flops)
    print('model_param_num:', params)
