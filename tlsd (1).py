import os
import concurrent.futures
import sys
import time
import subprocess
import shutil
from IPython.display import clear_output

params={}
for arg in sys.argv[1:]:
 if arg.startswith('--'):
  key_value=arg[len('--'):].split('=')
  if len(key_value)==2:
   key,value=key_value
   params[key]=value

subprocess.run(f'git clone -b master --single-branch https://github.com/AUTOMATIC1111/stable-diffusion-webui {params["sd_dir"]}',shell=True)

#subprocess.run(f'git clone -b master --single-branch https://github.com/anapnoe/stable-diffusion-webui-ux {params["sd_dir"]}',shell=True)


def task1():
 start_time = time.time()
 subprocess.run(f'git clone https://github.com/DominikDoom/a1111-sd-webui-tagcomplete {params["sd_dir"]}/extensions/a1111-sd-webui-tagcomplete',shell=True)
 subprocess.run(f'git clone https://github.com/Coyote-A/ultimate-upscale-for-automatic1111 {params["sd_dir"]}/extensions/ultimate-upscale',shell=True)
 subprocess.run(f'git clone https://github.com/hako-mikan/sd-webui-lora-block-weight {params["sd_dir"]}/extensions/sd-webui-lora-block-weight',shell=True)
 subprocess.run(f'git clone https://github.com/AIrjen/OneButtonPrompt {params["sd_dir"]}/extensions/OneButtonPromp',shell=True)
 subprocess.run(f'git clone https://github.com/adieyal/sd-dynamic-prompts {params["sd_dir"]}/extensions/sd-dynamic-prompts',shell=True)
 #subprocess.run(f'git clone https://github.com/zanllp/sd-webui-infinite-image-browsing {params["sd_dir"]}/extensions/sd-webui-infinite-image-browsing',shell=True)
 subprocess.run(f'git clone https://github.com/BlafKing/sd-civitai-browser-plus {params["sd_dir"]}/extensions/sd-civitai-browser-plus',shell=True)
 subprocess.run(f'git clone https://github.com/camenduru/tunnels {params["sd_dir"]}/extensions/tunnels',shell=True)

 if os.path.exists(f'{params["sd_dir"]}/embeddings'):
  shutil.rmtree(f'{params["sd_dir"]}/embeddings')
 subprocess.run(f'git clone https://huggingface.co/nolanaatama/embeddings {params["sd_dir"]}/embeddings',shell=True)
 
 end_time=time.time()
 print("\ntask1 spent:",end_time-start_time,"s")
def task2():
 start_time=time.time()
 subprocess.run(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth -d {params["sd_dir"]}/models/ESRGAN/ -o 4x-UltraSharp.pth',shell=True)
 subprocess.run(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/datasets/daasd/CN.csv/resolve/main/CN.csv -d {params["sd_dir"]}/extensions/a1111-sd-webui-tagcomplete/tags -o CN.csv',shell=True) 
 if(params['ckpt_url']):
  if(params['ckpt_url'].find('civitai')):    
   subprocess.run(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d {params["ckpt_dir"]} {params["ckpt_url"]}',shell=True)
  elif(params['ckpt_url'].find('huggingface')):    
   subprocess.run(f'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M -d {params["ckpt_dir"]} -o {params["ckpt_name"]} {params["ckpt_url"]}',shell=True)
  else:
   print('default')
 end_time=time.time() 
 print("\ntask2 spent:",end_time-start_time,"s")
 
def task3(): 
 start_time=time.time()
 subprocess.run(f"curl -Lo '{params['sd_dir']}/models/VAE/vae-ft-mse-840000-ema-pruned.safetensors' https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",shell=True)
 #subprocess.run(f"curl -Lo '{params['sd_dir']}/models/VAE/kl-f8-anime2.ckpt' https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/4c4f05104055c029ad577c18ac176462f0d1d7c1/vae/kl-f8-anime2.ckpt",shell=True)
 #subprocess.run(f"curl -Lo '{params['sd_dir']}/models/VAE/animevae.pt' https://huggingface.co/swl-models/animvae/resolve/main/animevae.pt",shell=True)
 end_time=time.time()
 print("\ntask3 spent:",end_time-start_time,"s")
def task4():
 start_time=time.time()
 subprocess.run("apt install libunwind8-dev -yqq",shell=True) 
 os.environ["LD_PRELOAD"]="libtcmalloc.so.4"
 os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
 subprocess.run("apt install sox ffmpeg libcairo2 libcairo2-dev",shell=True)
 end_time=time.time()
 print("\ntask4 spent:",end_time-start_time,"s")
def task5():
 start_time=time.time() 
 #subprocess.run("pip install xformers xformers==0.0.20",shell=True)
 subprocess.run("pip install -q xformers==0.0.20 triton==2.0.0 -U",shell=True) 
 end_time=time.time()
 print("\ntask5 spent:",end_time-start_time,"s")

executor=concurrent.futures.ThreadPoolExecutor(max_workers=5)
task1=executor.submit(task1)
task2=executor.submit(task2)
task3=executor.submit(task3)
task4=executor.submit(task4)
task5=executor.submit(task5)
concurrent.futures.wait([task5,task4,task3,task2,task1])

css_content = '''
  @media screen and (max-width: 600px) {
    .gradio-slider input[type="range"]{
      display: none;
    }
    .gradio-slider input[type="number"]{
      width: 18em;
    }
  }
  '''
with open(f'{params["sd_dir"]}/style.css', 'a') as cssFile:
      cssFile.write(css_content)


full_precision_str = params['user_arguments'] + " --multiple --disable-safe-unpickle --xformers --enable-insecure-extension-access --opt-sub-quad-attention --opt-channelslast --api --port 60088"
#full_precision_str=" --lowram --disable-safe-unpickle  --disable-console-progressbars --xformers --enable-insecure-extension-access --precision full --no-half --no-half-vae --opt-sub-quad-attention --opt-channelslast --api"

full_precision_str+=" --theme='dark'"

os.chdir(f'{params["sd_dir"]}')
subprocess.run(f"python launch.py {full_precision_str} --ui-settings-file {params['config_dir']} --styles-file {params['style_dir']} --lora-dir {params['lora_dir']} --ckpt-dir {params['ckpt_dir']} --ui-config-file {params['uiconfig_dir']}",shell=True)


