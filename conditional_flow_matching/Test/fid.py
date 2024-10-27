import torch

from conditional_flow_matching.Metric.fid import toFIDMetric

def test():
    random_sample_mash_params = torch.randn((10, 400), dtype=torch.float32, device = 'cuda:0')
    random_gt_mash_params = torch.randn((10, 32), dtype=torch.float32, device = 'cuda:0')

    random_sample_mash_params = random_sample_mash_params.reshape(10, -1)
    random_gt_mash_params = random_gt_mash_params.reshape(10, -1)

    fid = toFIDMetric(random_sample_mash_params, random_gt_mash_params)
    print(fid)
    return True
