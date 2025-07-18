import subprocess

from mash_diffusion.Demo.cfm_sampler import demo as demo_sample_cfm
from mash_diffusion.Method.time import getCurrentTime

if __name__ == "__main__":
    timestamp = getCurrentTime()

    out = subprocess.run(
        ["python", "create_sample_data.py", timestamp], capture_output=True, text=True
    )
    print(out.stdout)

    demo_sample_cfm(timestamp)
