from mash_diffusion.Demo.edm_trainer import demo as demo_train_edm
from mash_diffusion.Demo.cfm_trainer import demo as demo_train_cfm
from mash_diffusion.Demo.edm_sampler import demo as demo_sample_edm
from mash_diffusion.Demo.cfm_sampler import demo as demo_sample_cfm

if __name__ == "__main__":
    demo_train_edm()
    demo_train_cfm()
    demo_sample_edm()
    demo_sample_cfm()
