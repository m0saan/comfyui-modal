import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict

import modal
import modal.experimental


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install("comfy-cli")
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia")
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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def hf_download():
    """Download required models to cache volume if they don't already exist, then symlink to ComfyUI paths."""
    from huggingface_hub import hf_hub_download
    import shutil
    
    # git pull

    print("Updating comfyUI...")
    subprocess.run(
        "cd /root/comfy/ComfyUI && git pull && pip install -r requirements.txt",
        shell=True,
        check=True
    )
    print("ComfyUI update complete")
    
    # Install custom nodes from workflow dependencies
    print("Installing custom nodes from workflow...")
    subprocess.run(
        "comfy node install-deps --workflow=/root/video_upscale.json",
        shell=True,
        check=True
    )
    print("Custom nodes installation complete")

    print("Updating custom nodes...")
    subprocess.run(
        "comfy node update all",
        shell=True,
        check=True
    )
    print("Custom nodes update complete")

    
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
        # {
        #     "repo_id": "Kijai/sam2-safetensors",
        #     "filename": "sam2.1_hiera_base_plus.safetensors",
        #     "cache_dir": "/cache/models/loras",
        #     "comfy_dir": "/root/comfy/ComfyUI/models/loras",
        # }
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
    gpu="L40S",  # good starter GPU for inference
    volumes={"/cache": vol},  # mounts our cached models
)
@modal.concurrent(
    max_inputs=10
)  # required for UI startup process which runs several API calls concurrently
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
