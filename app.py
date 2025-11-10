import os
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import torchaudio
import gradio as gr
import json
import tempfile
from pathlib import Path
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

os.environ['DISABLE_FLASH_ATTN'] = "0"  # Enable flash-attn for faster inference
from SongBloom.models.songbloom.songbloom_pl import SongBloom_Sampler
from normalize_lyrics import clean_lyrics

NAME2REPO = {
    "songbloom_full_150s": "CypressYang/SongBloom",
    "songbloom_full_150s_dpo": "CypressYang/SongBloom",
    "songbloom_full_240s": "CypressYang/SongBloom_long",
}

# Global model cache
model_cache = {}
local_dir = "./models"

def download_model(model_name, progress=gr.Progress()):
    """Download model files from HuggingFace"""
    try:
        progress(0, desc="Starting download...")
        repo_id = NAME2REPO[model_name]
        
        progress(0.2, desc="Downloading config...")
        cfg_path = hf_hub_download(
            repo_id=repo_id, filename=f"{model_name}.yaml", local_dir=local_dir)
        
        progress(0.4, desc="Downloading model checkpoint...")
        ckpt_path = hf_hub_download(
            repo_id=repo_id, filename=f"{model_name}.pt", local_dir=local_dir)
        
        progress(0.6, desc="Downloading VAE config...")
        vae_cfg_path = hf_hub_download(
            repo_id="CypressYang/SongBloom", filename="stable_audio_1920_vae.json", local_dir=local_dir)
        
        progress(0.8, desc="Downloading VAE checkpoint...")
        vae_ckpt_path = hf_hub_download(
            repo_id="CypressYang/SongBloom", filename="autoencoder_music_dsp1920.ckpt", local_dir=local_dir)
        
        progress(0.9, desc="Downloading G2P vocab...")
        g2p_path = hf_hub_download(
            repo_id="CypressYang/SongBloom", filename="vocab_g2p.yaml", local_dir=local_dir)
        
        progress(1.0, desc="Download complete!")
        return "✅ Model downloaded successfully!"
    except Exception as e:
        return f"❌ Error downloading model: {str(e)}"

def load_config(cfg_file, parent_dir="./"):
    """Load configuration file"""
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda x: os.path.splitext(os.path.basename(x))[0])
    OmegaConf.register_new_resolver("load_yaml", lambda x: OmegaConf.load(x))
    OmegaConf.register_new_resolver("dynamic_path", lambda x: x.replace("???", parent_dir))
    
    file_cfg = OmegaConf.load(open(cfg_file, 'r')) if cfg_file is not None else OmegaConf.create()
    return file_cfg

def load_model(model_name, dtype_str="float32", device_str="cuda:0", progress=gr.Progress()):
    """Load model into memory"""
    global model_cache
    
    cache_key = f"{model_name}_{dtype_str}_{device_str}"
    if cache_key in model_cache:
        return "✅ Model already loaded!"
    
    try:
        progress(0, desc="Loading configuration...")
        cfg_path = f"{local_dir}/{model_name}.yaml"
        if not os.path.exists(cfg_path):
            return f"❌ Model not found. Please download it first."
        
        cfg = load_config(cfg_path, parent_dir=local_dir)
        cfg.max_dur = cfg.max_dur + 10
        
        progress(0.3, desc="Initializing model...")
        dtype = getattr(torch, dtype_str)
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        
        progress(0.6, desc="Loading weights...")
        model = SongBloom_Sampler.build_from_trainer(cfg, strict=False, dtype=dtype, device=device)
        model.set_generation_params(**cfg.inference)
        
        model_cache[cache_key] = {
            "model": model,
            "dtype": dtype,
            "device": device
        }
        
        progress(1.0, desc="Model loaded!")
        return f"✅ Model loaded successfully on {device}!"
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def generate_song(model_name, lyrics, prompt_audio, n_samples, dtype_str, device_str, progress=gr.Progress()):
    """Generate song from lyrics and prompt audio"""
    global model_cache
    
    cache_key = f"{model_name}_{dtype_str}_{device_str}"
    if cache_key not in model_cache:
        return None, "❌ Please load the model first!"
    
    try:
        model_data = model_cache[cache_key]
        model = model_data["model"]
        dtype = model_data["dtype"]
        
        if prompt_audio is None:
            return None, "❌ Please upload a prompt audio file!"
        
        if not lyrics.strip():
            return None, "❌ Please enter lyrics!"
        
        progress(0, desc="Processing prompt audio...")
        # Load and process prompt audio
        prompt_wav, sr = torchaudio.load(prompt_audio)
        if sr != model.sample_rate:
            prompt_wav = torchaudio.functional.resample(prompt_wav, sr, model.sample_rate)
        prompt_wav = prompt_wav.mean(dim=0, keepdim=True).to(dtype)
        prompt_wav = prompt_wav[..., :10*model.sample_rate]
        
        # Generate songs
        output_files = []
        for i in range(n_samples):
            progress((i + 1) / n_samples, desc=f"Generating sample {i+1}/{n_samples}...")
            wav = model.generate(lyrics, prompt_wav)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".flac")
            torchaudio.save(temp_file.name, wav[0].cpu().float(), model.sample_rate)
            output_files.append(temp_file.name)
        
        progress(1.0, desc="Generation complete!")
        
        if len(output_files) == 1:
            return output_files[0], "✅ Song generated successfully!"
        else:
            return output_files, "✅ Songs generated successfully!"
    
    except Exception as e:
        return None, f"❌ Error generating song: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="SongBloom - Song Generation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎵 SongBloom: Song Generation Web UI
    Generate full-length songs with lyrics and style prompts using SongBloom models.
    """)
    
    with gr.Tab("Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Settings")
                model_dropdown = gr.Dropdown(
                    choices=list(NAME2REPO.keys()),
                    value="songbloom_full_150s",
                    label="Model",
                    info="Choose the model version"
                )
                dtype_dropdown = gr.Dropdown(
                    choices=["float32", "bfloat16"],
                    value="bfloat16",
                    label="Data Type",
                    info="Use bfloat16 for GPUs with low VRAM"
                )
                device_dropdown = gr.Dropdown(
                    choices=["cuda:0", "cpu"],
                    value="cuda:0" if torch.cuda.is_available() else "cpu",
                    label="Device"
                )
                
                gr.Markdown("### Input")
                lyrics_input = gr.Textbox(
                    label="Lyrics",
                    value="[intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] [intro] , [verse] Are you going to Scarborough Fair. Parsley sage rosemary and thyme. Remember me to one who lives there. He once was a true love of mine , [chorus] Tell him to make me a cambric shirt. Parsley sage rosemary and thyme. Without no seams nor needlework. Then he'll be a true love of mine , [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] , [chorus] Tell him to reap it with a sickle of leather. Parsley sage rosemary and thyme. And gather it all in a bunch of heather. Then he'll be a true love of mine , [verse] Are you going to Scarborough Fair. Parsley sage rosemary and thyme. Remember me to one who lives there. He once was a true love of mine , [outro] [outro] [outro] [outro] [outro] [outro]",
                    placeholder="Enter your lyrics here...\nSee docs/lyric_format.md for formatting details",
                    lines=8
                )
                prompt_audio = gr.Audio(
                    label="Style Prompt Audio (10s, 48kHz recommended)",
                    type="filepath"
                )
                n_samples = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Number of Samples",
                    info="How many variations to generate"
                )
                
                generate_btn = gr.Button("🎵 Generate Song", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                output_audio = gr.Audio(label="Generated Song")
                status_output = gr.Textbox(label="Status", lines=2)
    
    with gr.Tab("Model Management"):
        gr.Markdown("""
        ### Download and Load Models
        First download the model files, then load them into memory before generating.
        """)
        
        with gr.Row():
            with gr.Column():
                download_model_dropdown = gr.Dropdown(
                    choices=list(NAME2REPO.keys()),
                    value="songbloom_full_150s",
                    label="Select Model to Download"
                )
                download_btn = gr.Button("📥 Download Model", variant="primary")
                download_status = gr.Textbox(label="Download Status", lines=3)
            
            with gr.Column():
                load_model_dropdown = gr.Dropdown(
                    choices=list(NAME2REPO.keys()),
                    value="songbloom_full_150s",
                    label="Select Model to Load"
                )
                load_dtype = gr.Dropdown(
                    choices=["float32", "bfloat16"],
                    value="bfloat16",
                    label="Data Type"
                )
                load_device = gr.Dropdown(
                    choices=["cuda:0", "cpu"],
                    value="cuda:0" if torch.cuda.is_available() else "cpu",
                    label="Device"
                )
                load_btn = gr.Button("🔄 Load Model", variant="primary")
                load_status = gr.Textbox(label="Load Status", lines=3)
    
    with gr.Tab("Info"):
        gr.Markdown("""
        ### About SongBloom
        
        SongBloom is a novel framework for full-length song generation that leverages an interleaved paradigm 
        of autoregressive sketching and diffusion-based refinement.
        
        **Available Models:**
        - `songbloom_full_150s`: 2B params, up to 2m30s generation
        - `songbloom_full_150s_dpo`: DPO post-trained version
        - `songbloom_full_240s`: 2B params, up to 4m generation
        
        **Usage Tips:**
        1. Download your desired model in the "Model Management" tab
        2. Load the model into memory
        3. Upload a 10-second style prompt audio (48kHz recommended)
        4. Enter your lyrics (see docs/lyric_format.md for formatting)
        5. Click "Generate Song"
        
        **System Requirements:**
        - CUDA-capable GPU recommended
        - For low VRAM GPUs (e.g., RTX 4090), use bfloat16 dtype
        
        [Paper](https://arxiv.org/abs/2506.07634) | 
        [Models](https://huggingface.co/CypressYang/SongBloom) | 
        [Demo](https://cypress-yang.github.io/SongBloom_demo)
        """)
    
    # Event handlers
    download_btn.click(
        fn=download_model,
        inputs=[download_model_dropdown],
        outputs=[download_status]
    )
    
    load_btn.click(
        fn=load_model,
        inputs=[load_model_dropdown, load_dtype, load_device],
        outputs=[load_status]
    )
    
    generate_btn.click(
        fn=generate_song,
        inputs=[model_dropdown, lyrics_input, prompt_audio, n_samples, dtype_dropdown, device_dropdown],
        outputs=[output_audio, status_output]
    )

if __name__ == "__main__":
    os.makedirs(local_dir, exist_ok=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
