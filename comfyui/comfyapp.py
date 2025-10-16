import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict

import modal
import modal.experimental
COMFY_DIR = "/root/comfy/ComfyUI"
CIVITAI_API_TOKEN = "165de85d4ecf11f5ce04786082a06a94"
# manual custom node install, if comfy-cli doesn't work
def git_install_custom_node(repo_id: str, recursive: bool = False, install_reqs: bool = False):
    custom_node = repo_id.split("/")[-1]
    
    command = f"git clone https://github.com/{repo_id}"

    if recursive:
        command += " --recursive"

    command += f" {COMFY_DIR}/custom_nodes/{custom_node}"

    if install_reqs:
        command += f" && uv pip install --system --compile-bytecode -r {COMFY_DIR}/custom_nodes/{custom_node}/requirements.txt"

    return command


# download from civitai to cache and create symlink
def civitai_download(local_dir: str, filename: str, url: str):
    """Download from Civitai to cache directory and create symlink to ComfyUI models directory."""
    import subprocess
    from pathlib import Path
    
    cache_path = Path(f"/cache/models/{local_dir}") / filename
    comfy_path = Path(f"{COMFY_DIR}/models/{local_dir}") / filename
    
    # Create cache directory if it doesn't exist
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists in cache
    if cache_path.exists():
        print(f"Model already exists in cache, skipping download: {cache_path}")
    else:
        # Download model using comfy-cli to cache
        print(f"Downloading {filename} from Civitai to cache...")
        download_cmd = (
            f"comfy --skip-prompt model download --url '{url}' "
            f"--relative-path '{cache_path.parent}' "
            f"--filename '{filename}' "
            f"--set-civitai-api-token {CIVITAI_API_TOKEN}"
        )
        subprocess.run(download_cmd, shell=True, check=True)
        print(f"Successfully cached: {cache_path}")
    
    # Create ComfyUI directory if it doesn't exist
    comfy_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create symlink from ComfyUI path to cache (if not already exists)
    if not comfy_path.exists():
        print(f"Creating symlink: {comfy_path} -> {cache_path}")
        comfy_path.symlink_to(cache_path)
    else:
        print(f"Symlink already exists: {comfy_path}")



image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git")
    .run_commands("pip install --upgrade pip")
    .pip_install("uv")
    .run_commands("uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands("uv pip install --system --compile-bytecode comfy-cli==1.3.6")
    .run_commands("comfy --skip-prompt install --nvidia")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0") # required for several custom nodes on Linux
    .run_commands("pip install --upgrade comfy-cli")
    .run_commands("comfy --version")
    .add_local_file(
        Path(__file__).parent / "video_upscale.json", "/root/video_upscale.json", copy=True
    )
    .uv_pip_install(
        "accelerate",
        "huggingface-hub[hf-transfer]",
        "Pillow",
        "safetensors",
        "transformers",
        "torch==2.8.0",
        extra_options="--index-strategy unsafe-best-match",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands("comfy node install ComfyUI-WanVideoWrapper")
    .run_commands("comfy node install rgthree-comfy")
    .run_commands("comfy node install comfyui-florence2")
    .run_commands("comfy node install comfyui-frame-interpolation")
    .run_commands("comfy node install comfyui-segment-anything-2")
    .run_commands("comfy node install comfyui_layerstyle")
    .run_commands("cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/un-seen/comfyui-tensorops")
    .run_commands(git_install_custom_node("ssitu/ComfyUI_UltimateSDUpscale", recursive=True))
    .run_commands(git_install_custom_node("kijai/ComfyUI-KJNodes", install_reqs=True, recursive=True))
    .run_commands(git_install_custom_node("Kosinkadink/ComfyUI-VideoHelperSuite", install_reqs=True, recursive=True))
    .run_commands(git_install_custom_node("Fannovel16/comfyui_controlnet_aux"))
    .run_commands("python -m pip install -U scikit-image")
    .run_commands("comfy node install comfyui-easy-use")
    .run_commands("comfy node install basic_data_handling")
    .run_commands("comfy node install comfyui_essentials")
    .run_commands("comfy node install seedvr2_videoupscaler")
    .run_commands(git_install_custom_node("ClownsharkBatwing/RES4LYF"))
    .run_commands("comfy node install comfyui-propost@1.1.2")
    .run_commands("comfy node install comfyui_face_parsing")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
# comfy node install comfyui-easy-use
# comfy node install basic_data_handling
# comfy node install comfyui_essentials
# comfy node install seedvr2_videoupscaler
# https://github.com/ClownsharkBatwing/RES4LYF


def hf_download():
    """Download required models to cache volume if they don't already exist, then symlink to ComfyUI paths."""
    from huggingface_hub import hf_hub_download
    import shutil
    
    
    # Define models to download with their HuggingFace repo info and ComfyUI paths
    models_to_check = [
        {
            "repo_id": "Wan-AI/Wan2.2-T2V-A14B",
            "filename": "models_t5_umt5-xxl-enc-bf16.pth",
            "cache_dir": "/cache/models/clip",
            "comfy_dir": "/root/comfy/ComfyUI/models/clip",
        },
        {
            "repo_id": "Wan-AI/Wan2.2-T2V-A14B",
            "filename": "Wan2.1_VAE.pth",
            "cache_dir": "/cache/models/vae",
            "comfy_dir": "/root/comfy/ComfyUI/models/vae",
        },
        {
            "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "filename": "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors",
            "local_filename": "wan2.2_t2v_low_noise_14B_fp16.safetensors",
            "cache_dir": "/cache/models/diffusion_models",
            "comfy_dir": "/root/comfy/ComfyUI/models/diffusion_models",
        },
        {
            "repo_id": "lllyasviel/Annotators",
            "filename": "RealESRGAN_x4plus.pth",
            "cache_dir": "/cache/models/upscalers",
            "comfy_dir": "/root/comfy/ComfyUI/models/upscalers",
        },
        {
            "repo_id": "Kijai/WanVideo_comfy",
            "filename": "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
            "cache_dir": "/cache/models/loras",
            "comfy_dir": "/root/comfy/ComfyUI/models/loras",
        },
        {
            "repo_id": "alibaba-pai/Wan2.2-Fun-Reward-LoRAs",
            "filename": "Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors",
            "cache_dir": "/cache/models/loras",
            "comfy_dir": "/root/comfy/ComfyUI/models/loras",
        },
        {
            # https://huggingface.co/aidiffuser/Qwen-Image-Edit-2509/resolve/main/Qwen-Image-Edit-2509_fp8_e4m3fn.safetensors
            "repo_id": "aidiffuser/Qwen-Image-Edit-2509",
            "filename": "Qwen-Image-Edit-2509_fp8_e4m3fn.safetensors",
            "cache_dir": "/cache/models/diffusion_models",
            "comfy_dir": "/root/comfy/ComfyUI/models/diffusion_models",
        },
        {
            # https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
            "repo_id": "Comfy-Org/Qwen-Image_ComfyUI",
            "filename": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "cache_dir": "/cache/models/clip",
            "comfy_dir": "/root/comfy/ComfyUI/models/clip",
        },
        {
            #https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors
            "repo_id": "lightx2v/Qwen-Image-Lightning",
            "filename": "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
            "cache_dir": "/cache/models/loras",
            "comfy_dir": "/root/comfy/ComfyUI/models/loras",
        },
        {
            "repo_id": "RunDiffusion/Juggernaut-XL-v9",
            "filename": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
            "cache_dir": "/cache/models/checkpoints",
            "comfy_dir": "/root/comfy/ComfyUI/models/checkpoints",
        },
        {
            "repo_id": "xinsir/controlnet-tile-sdxl-1.0",
            "filename": "diffusion_pytorch_model.safetensors",
            "cache_dir": "/cache/models/controlnet",
            "comfy_dir": "/root/comfy/ComfyUI/models/controlnet",
        },
        {
            "repo_id": "ByteDance/SDXL-Lightning",
            "filename": "sdxl_lightning_8step_lora.safetensors",
            "cache_dir": "/cache/models/loras",
            "comfy_dir": "/root/comfy/ComfyUI/models/loras",
        },
        {
            "repo_id": "labai-llc/skin-fix",
            "filename": "skin_realism-248951.safetensors",
            "cache_dir": "/cache/models/loras",
            "comfy_dir": "/root/comfy/ComfyUI/models/loras",
        },
        {
            "repo_id": "labai-llc/skin-fix",
            "filename": "better_freckles_sdxl.safetensors",
            "cache_dir": "/cache/models/loras",
            "comfy_dir": "/root/comfy/ComfyUI/models/loras",
        },
        {
            "repo_id": "labai-llc/skin-fix",
            "filename": "RealVisXL_V5.0_fp16.safetensors",
            "cache_dir": "/cache/models/checkpoints",
            "comfy_dir": "/root/comfy/ComfyUI/models/checkpoints",
        },
        {
            "repo_id": "labai-llc/skin-fix",
            "filename": "8xNMKDFaces160000G_v10.pt",
            "cache_dir": "/cache/models/upscale",
            "comfy_dir": "/root/comfy/ComfyUI/models/upscale_models",
        }
    ]
    
    for model in models_to_check:
        # Use local_filename if specified, otherwise extract from filename
        local_filename = model.get("local_filename", Path(model["filename"]).name)
        cache_path = Path(model["cache_dir"]) / local_filename
        comfy_path = Path(model["comfy_dir"]) / local_filename
        
        # Create cache directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists in cache
        if cache_path.exists():
            print(f"Model already exists in cache, skipping download: {cache_path}")
        else:
            # Download model using HuggingFace Hub (with HF transfer for faster downloads)
            print(f"Downloading {model['repo_id']}/{model['filename']} using HuggingFace Hub...")
            downloaded_path = hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"],
                cache_dir="/cache/huggingface",
                local_dir=None,
            )
            print(f"Downloaded to: {downloaded_path}")
            
            # Copy the downloaded file to our cache location
            print(f"Copying to cache: {cache_path}")
            shutil.copy2(downloaded_path, cache_path)
            print(f"Successfully cached: {cache_path}")
        
        # Create ComfyUI directory if it doesn't exist
        comfy_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create symlink from ComfyUI path to cache (if not already exists)
        if not comfy_path.exists():
            print(f"Creating symlink: {comfy_path} -> {cache_path}")
            comfy_path.symlink_to(cache_path)
        else:
            print(f"Symlink already exists: {comfy_path}")


vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    image.run_function(
        hf_download,
        # persist the HF cache to a Modal Volume so future runs don't re-download models
        volumes={"/cache": vol},
    )
    .add_local_file(
        Path(__file__).parent / "workflow_api.json", "/root/workflow_api.json"
    )
)


app = modal.App(name="example-comfyapp", image=image)


@app.function(
    max_containers=1,  # limit interactive session to 1 container
    gpu="A100-40GB",  # good starter GPU for inference
    volumes={"/cache": vol},  # mounts our cached models
    timeout=900,
)
@modal.concurrent(
    max_inputs=10
)  # required for UI startup process which runs several API calls concurrently
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
