import subprocess
import concurrent.futures
from IPython.display import clear_output
import os
import sys

params = {}
for arg in sys.argv[1:]:
    if arg.startswith('--'):
        key_value = arg[len('--'):].split('=')
        if len(key_value) == 2:
            key, value = key_value
            params[key] = value

subprocess.run(f'git clone https://github.com/Lucysck/test1 {params["dir"]}', shell=True)

def run_git():
    subprocess.run(f'cp -rf /content/drive/MyDrive/sd/* {params["dir"]}', shell=True)

def run_aria2c():
    subprocess.run(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/weo1101/111/resolve/main/chilloutmix_NiPrunedFp32Fix-inpainting.inpainting.safetensors -d {params["dir"]}/models/Stable-diffusion -o chilloutmix_NiPrunedFp32Fix-inpainting.inpainting.safetensors', shell=True)

def run_code():
    subprocess.run(f"wget -O {params['dir']}/libtcmalloc_minimal.so.4 https://huggingface.co/Vanwise/sd-colab/resolve/main/libtcmalloc_minimal.so.4", shell=True)
    os.environ['LD_PRELOAD'] = f"{params['dir']}/libtcmalloc_minimal.so.4"

executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
task1 = executor.submit(run_git)
task2 = executor.submit(run_aria2c)
task3 = executor.submit(run_code)
concurrent.futures.wait([task1, task2, task3])

subprocess.run(f"wget -O {params['dir']}/config.json https://huggingface.co/spaces/weo1101/111/resolve/main/config.json", shell=True)

import threading
import time

def clear_output_pro():
    clear_output()
    print("Clearing output...")
    threading.Timer(30, clear_output_pro).start()

threading.Timer(360, clear_output_pro).start()

os.chdir(f'{params["dir"]}')
clear_output()

base_arg = "--disable-safe-unpickle --opt-sdp-attention --no-half-vae --enable-insecure-extension --theme=dark --lowram  --listen --xformers"

if params["share"] == "True":
    base_arg += " --share"

if params["ngrok"]:
    base_arg += f' --ngrok={params["ngrok"]} --ngrok-region="auto"'

if params["cloudflared"] == "True":
    base_arg += f" --cloudflared"

if params["localhostrun"] == "True":
    base_arg += f" --localhostrun"

if params["remotemoe"] == "True":
    base_arg += f" --remotemoe"

subprocess.run(f"python launch.py {base_arg}", shell=True)