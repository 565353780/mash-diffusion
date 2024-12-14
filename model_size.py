import torch
from conditional_flow_matching.Model.mash_latent_net import MashLatentNet
from conditional_flow_matching.Method.model_size import getModelFLOPSAndParamsNum

if __name__ == "__main__":
    model = MashLatentNet(
        n_latents=400,
        mask_degree=3,
        sh_degree=2,
        embed_dim=1024,
        context_dim=1024,
        n_heads=4,
        d_head=256,
        depth=24)
    inputs = (torch.randn(1, 400, 25), torch.randn(1, 1, 1024), torch.randn(1))

    flops, params = getModelFLOPSAndParamsNum(model, inputs)

    print('model_FLOPs:', flops)
    print('model_param_num:', params)
