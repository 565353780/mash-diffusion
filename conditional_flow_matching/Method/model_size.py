import torch
from thop import profile
from thop import clever_format

def getModelParameterNum(model: torch.nn.Module) -> int:
    model_param_num = sum(p.numel() for p in model.parameters())
    return model_param_num

def getModelFLOPSAndParamsNum(model: torch.nn.Module, inputs: tuple) -> bool:
    flops, params = profile(model, inputs=inputs)

    flops, params = clever_format([flops, params], '%.3f')
    return flops, params
