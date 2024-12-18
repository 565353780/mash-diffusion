import torch
from conditional_flow_matching.Model.mash_latent_net import MashLatentNet
from conditional_flow_matching.Method.model_size import getModelFLOPSAndParamsNum

if __name__ == "__main__":
    batch_size = 1
    embed_dim = 1536
    context_num = 1398
    context_dim = 1536

    model = MashLatentNet(
        n_latents=400,
        mask_degree=3,
        sh_degree=2,
        embed_dim=embed_dim,
        context_dim=context_dim,
        n_heads=16,
        d_head=96,
        depth=24)
    inputs = (torch.randn(batch_size, 400, 25), torch.randn(batch_size, context_num, context_dim), torch.randn(batch_size))

    flops, params = getModelFLOPSAndParamsNum(model, inputs)

    print('model_FLOPs:', flops)
    print('model_param_num:', params)
