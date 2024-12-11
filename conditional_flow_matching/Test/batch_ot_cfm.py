import sys
sys.path.append('../ma-sh/')

import numpy as np

from ma_sh.Method.random_mash import sampleRandomMashParams

from conditional_flow_matching.Module.batch_ot_cfm import BatchExactOptimalTransportConditionalFlowMatcher

def test():
    sigma = 0.0
    target_dim = [6, 7, 8]

    x0 = sampleRandomMashParams(400, 3, 2, 10, 'cpu', False)

    permute_idxs = np.random.permutation(x0.shape[0])
    x1 = x0[permute_idxs] * 4.0

    batch_ot_cfm = BatchExactOptimalTransportConditionalFlowMatcher(sigma, target_dim)

    t, xt, ut = batch_ot_cfm.sample_location_and_conditional_flow(x0, x1)

    print(t)
    print(t.shape)
    print(xt.shape)
    print(ut.shape)

    return True
