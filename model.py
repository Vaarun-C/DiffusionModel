from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
import modal
import torch
import base64
from io import BytesIO
import time
import os
import hashlib
import json
import pickle

# Define the Modal app
app = modal.App("stable-diffusion-2-1-cached-watermarked")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch==2.0.1",
        "torchvision==0.15.2",
        "huggingface-hub==0.17.3",
        "diffusers==0.21.4",
        "transformers==4.34.1",
        "accelerate==0.23.0",
        "safetensors==0.4.0",
        "Pillow==10.0.1",
        "numpy==1.24.3",
        "xformers==0.0.22",
        "fastapi[standard]==0.104.1",
        "detoxify==0.5.2",
        "better-profanity==0.7.0",
        "gradio==4.7.1"
    ])
)

# Global variables for model and cache storage
MODEL_CACHE = "/cache"
IMAGE_CACHE = "/image_cache"

def create_cache_key(
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int = None,
    watermark_text: str = None,
    watermark_opacity: float = None,
    watermark_position: str = None
) -> str:
    """Create a unique cache key from generation parameters including watermark settings"""
    cache_params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "seed": seed,
        "watermark_text": watermark_text,
        "watermark_opacity": watermark_opacity,
        "watermark_position": watermark_position
    }
    
    cache_string = json.dumps(cache_params, sort_keys=True)
    return hashlib.sha256(cache_string.encode()).hexdigest()

def add_watermark(
    image: Image.Image, 
    text: str = "AI Generated", 
    position: str = "bottom-right",
    opacity: float = 0.7,
    font_size: int = None,
    font_color: tuple = (255, 255, 255),
    margin: int = 10
) -> Image.Image:
    watermarked = image.copy()

    if font_size is None:
        font_size = max(12, min(image.width, image.height) // 25)
        
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(watermarked)
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    img_width, img_height = image.size
    
    if position == "bottom-right":
        x = img_width - text_width - margin
        y = img_height - text_height - margin
    elif position == "bottom-left":
        x = margin
        y = img_height - text_height - margin
    elif position == "top-right":
        x = img_width - text_width - margin
        y = margin
    elif position == "top-left":
        x = margin
        y = margin
    elif position == "center":
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2
    else:
        # Default to bottom-right
        x = img_width - text_width - margin
        y = img_height - text_height - margin
    

    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    text_color_with_opacity = font_color + (int(255 * opacity),)
    overlay_draw.text((x, y), text, font=font, fill=text_color_with_opacity)
    
    # Composite the overlay onto the original image
    watermarked = Image.alpha_composite(watermarked.convert('RGBA'), overlay)
    
    return watermarked

def load_safety_models():
    """Called once per container"""
    global safety_detector
    
    from detoxify import Detoxify
    safety_detector = Detoxify('original')

def check_prompt_safety(prompt: str, negative_prompt: str = ""):
    full_text = f"{prompt} {negative_prompt}".strip()
    
    # Basic keyword filtering
    nsfw_keywords = [
        'nude', 'adult', 'nsfw', 'explicit',
        'violent', 'gore', 'blood', 'death', 'kill', 'murder',
        'drugs', 'cocaine', 'heroin', 'meth'
    ]
    
    text_lower = full_text.lower()
    for keyword in nsfw_keywords:
        if keyword in text_lower:
            return False, f"Contains restricted keyword: {keyword}", {"keyword_match": keyword}
    
    # AI-based detection
    if 'safety_detector' in globals() and safety_detector is not None:
        try:
            scores = safety_detector.predict(full_text)
            
            thresholds = {
                'toxicity': 0.7,
                'severe_toxicity': 0.5,
                'obscene': 0.7,
                'threat': 0.7,
                'insult': 0.8,
                'identity_attack': 0.7
            }
            
            # Check each category
            for category, threshold in thresholds.items():
                if scores.get(category, 0) > threshold:
                    return False, f"High {category} score: {scores[category]:.3f}", scores
            
            return True, "Content appears safe", scores
            
        except Exception as e:
            print(f"Safety check error: {e}")
            return True, "Safety check unavailable", {}
    
    return True, "Basic safety check passed", {}

def save_to_cache(cache_key: str, result_data: dict):
    cache_dir = f"{IMAGE_CACHE}/images"
    metadata_dir = f"{IMAGE_CACHE}/metadata"
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    image_path = f"{cache_dir}/{cache_key}.pkl"
    with open(image_path, 'wb') as f:
        pickle.dump(result_data, f)
    
    # Save metadata for easier inspection
    metadata = {
        "cache_key": cache_key,
        "prompt": result_data.get("prompt"),
        "generation_time": result_data.get("generation_time"),
        "parameters": result_data.get("parameters"),
        "watermark_settings": result_data.get("watermark_settings"),
        "cached_at": time.time()
    }
    
    metadata_path = f"{metadata_dir}/{cache_key}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved to cache: {cache_key}")

def load_from_cache(cache_key: str) -> dict:
    cache_path = f"{IMAGE_CACHE}/images/{cache_key}.pkl"
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            result_data = pickle.load(f)
        
        print(f"Loaded from cache: {cache_key}")

        result_data["from_cache"] = True
        result_data["cache_key"] = cache_key
        return result_data
    
    return None

@app.function(
    image=image,
    gpu="A10G",
    memory=16000,
    timeout=300,
    scaledown_window=300,
    volumes={
        MODEL_CACHE: modal.Volume.from_name("sd-model-cache", create_if_missing=True),
        IMAGE_CACHE: modal.Volume.from_name("sd-image-cache", create_if_missing=True)
    }
)
def generate_image_cached(
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = None,
    force_regenerate: bool = False,
    enable_safety_filter: bool = True,
    strict_safety: bool = False,
    add_watermark_flag: bool = True,
    watermark_text: str = "AI Generated",
    watermark_position: str = "bottom-right",
    watermark_opacity: float = 0.7,
    watermark_font_size: int = None,
    watermark_font_color: tuple = (255, 255, 255),
) -> dict:
    
    if 'safety_detector' not in globals():
        load_safety_models()
    
    cache_key = create_cache_key(
        prompt, negative_prompt, num_inference_steps, 
        guidance_scale, width, height, seed,
        watermark_text if add_watermark_flag else None,
        watermark_opacity if add_watermark_flag else None,
        watermark_position if add_watermark_flag else None
    )
    
    print(f"Cache key: {cache_key[:16]}...")
    
    if not force_regenerate:
        cached_result = load_from_cache(cache_key)
        if cached_result:
            print("Cache hit! Returning cached image")
            return cached_result
    
    print(f"Cache miss. Generating new image for: '{prompt}'")
    start_time = time.time()
    
    if enable_safety_filter:
        is_safe, reason, safety_scores = check_prompt_safety(prompt, negative_prompt)
        
        if not is_safe:
            print(f"Safety check failed: {reason}")
            
            if strict_safety:
                return {
                    "error": "Content policy violation",
                    "reason": reason,
                    "safety_scores": safety_scores,
                    "prompt": prompt,
                    "generation_time": 0,
                    "from_cache": False
                }
            else:
                print(f"Warning: {reason} - Continuing with content filter")
                negative_prompt = f"{negative_prompt}, nsfw, explicit, inappropriate content"
        
        print(f"Safety check: {reason}")
    
    model_id = "stabilityai/stable-diffusion-2-1"
    cache_dir = f"{MODEL_CACHE}/models"
    os.makedirs(cache_dir, exist_ok=True)
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    
    # Enable memory optimizations
    pipe.enable_attention_slicing()
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("XFormers memory efficient attention enabled")
    except Exception as e:
        print(f"XFormers not available, using default attention: {e}")
        try:
            pipe.enable_model_cpu_offload()
            print("Model CPU offload enabled")
        except Exception as e2:
            print(f"Model CPU offload also failed: {e2}")
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    
    try:
        with torch.autocast("cuda"):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )
            
            image = result.images[0]
            
    except Exception as e:
        print(f"Error during generation: {e}")
        image = Image.new('RGB', (width, height), color='red')
    

    watermark_settings = None
    if add_watermark_flag:
        print(f"Adding watermark: '{watermark_text}' at {watermark_position}")
        watermark_settings = {
            "text": watermark_text,
            "position": watermark_position,
            "opacity": watermark_opacity,
            "font_size": watermark_font_size,
            "font_color": watermark_font_color
        }
        
        try:
            image = add_watermark(
                image=image,
                text=watermark_text,
                position=watermark_position,
                opacity=watermark_opacity,
                font_size=watermark_font_size,
                font_color=watermark_font_color
            )
            print("Watermark added successfully")
        except Exception as e:
            print(f"Warning: Failed to add watermark: {e}")
            # Continue without watermark rather than failing completely
        
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    result_data = {
        "image_base64": img_base64,
        "generation_time": generation_time,
        "prompt": prompt,
        "seed": seed,
        "parameters": {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height
        },
        "watermark_settings": watermark_settings,
        "from_cache": False,
        "cache_key": cache_key
    }
    
    save_to_cache(cache_key, result_data)
    return result_data

@app.function(
    image=image,
    volumes={IMAGE_CACHE: modal.Volume.from_name("sd-image-cache", create_if_missing=True)}
)
def get_cache_stats() -> dict:
    """Get statistics about the cache"""
    
    cache_dir = f"{IMAGE_CACHE}/images"
    metadata_dir = f"{IMAGE_CACHE}/metadata"
    
    if not os.path.exists(cache_dir):
        return {"total_cached_images": 0, "cache_size_mb": 0, "cached_prompts": []}
    
    # Count cached images
    cached_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
    total_cached = len(cached_files)
    
    # Calculate cache size
    total_size = 0
    for filename in cached_files:
        file_path = os.path.join(cache_dir, filename)
        total_size += os.path.getsize(file_path)
    
    cache_size_mb = total_size / (1024 * 1024)
    
    # Get sample of cached prompts
    cached_prompts = []
    if os.path.exists(metadata_dir):
        metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')][:10]  # Sample first 10
        
        for filename in metadata_files:
            try:
                with open(os.path.join(metadata_dir, filename), 'r') as f:
                    metadata = json.load(f)
                    cached_prompts.append({
                        "prompt": metadata.get("prompt", "Unknown"),
                        "cached_at": metadata.get("cached_at"),
                        "generation_time": metadata.get("generation_time"),
                        "watermark_settings": metadata.get("watermark_settings")
                    })
            except:
                continue
    
    return {
        "total_cached_images": total_cached,
        "cache_size_mb": round(cache_size_mb, 2),
        "cached_prompts": cached_prompts
    }

@app.function(
    image=image,
    volumes={IMAGE_CACHE: modal.Volume.from_name("sd-image-cache", create_if_missing=True)}
)
def clear_cache() -> dict:
    import shutil
    
    cache_dir = f"{IMAGE_CACHE}/images"
    metadata_dir = f"{IMAGE_CACHE}/metadata"
    
    deleted_count = 0
    
    if os.path.exists(cache_dir):
        files = os.listdir(cache_dir)
        deleted_count = len(files)
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
    
    if os.path.exists(metadata_dir):
        shutil.rmtree(metadata_dir)
        os.makedirs(metadata_dir, exist_ok=True)
    
    return {"message": f"Cache cleared. Deleted {deleted_count} cached images."}

def safe_filename(prompt: str, index: int) -> str:
    safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')[:50]
    return f"modal_output_{index:02d}_{safe_prompt}.png"

@app.function(image=image)
def multiple_prompts_cached(
    prompts: list[str],
    negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int = None,
    force_regenerate: bool = False,
    enable_safety_filter: bool = True,
    strict_safety: bool = False,
    add_watermark_flag: bool = True,
    watermark_text: str = "AI Generated",
) -> dict:
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}: {prompt}")
        
        result = generate_image_cached.remote(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            width=int(width),
            height=int(height),
            seed=None if seed in (-1, None) else int(seed),
            add_watermark_flag=bool(add_watermark_flag),
            watermark_text=watermark_text,
            force_regenerate=bool(force_regenerate),
            enable_safety_filter=bool(enable_safety_filter),
            strict_safety=bool(strict_safety),
        )
        
        results.append({
            "prompt": prompt,
            "result": result,
            "index": i + 1
        })
    
    return {"results": results, "total_prompts": len(prompts)}

@app.function(
    image=image,
    max_containers=1,           # sticky sessions
    scaledown_window=60 * 20,   # auto-shutdown
    min_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def gradio_app():
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from PIL import Image as PILImage
    import io, base64

    api = FastAPI()

    def generate_image_ui(
        prompt,
        negative_prompt,
        steps,
        guidance,
        width,
        height,
        seed,
        add_watermark_flag,
        watermark_text,
        force_regenerate,
        enable_safety_filter,
        strict_safety,
        batch_prompts_str,
    ):
        # If user supplied batch prompts, ignore single-prompt path:
        lines = [l.strip() for l in batch_prompts_str.splitlines() if l.strip()]
        if lines:
            batch_result = multiple_prompts_cached.remote(
                prompts=lines,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                width=int(width),
                height=int(height),
                seed=None if seed in (-1, None) else int(seed),
                add_watermark_flag=bool(add_watermark_flag),
                watermark_text=watermark_text,
                force_regenerate=bool(force_regenerate),
                enable_safety_filter=bool(enable_safety_filter),
                strict_safety=bool(strict_safety)
            )
            images = []
            for entry in batch_result["results"]:
                b64 = entry["result"]["image_base64"]
                img = PILImage.open(io.BytesIO(base64.b64decode(b64)))
                images.append(img)
            info = f"Processed {len(images)} prompts — {sum(1 for e in batch_result['results'] if e['result'].get('from_cache'))} cached"
            return None, images, info

        # Single-prompt path:
        result = generate_image_cached.remote(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            seed=None if seed in (-1, None) else int(seed),
            add_watermark_flag=bool(add_watermark_flag),
            watermark_text=watermark_text,
            force_regenerate=bool(force_regenerate),
            enable_safety_filter=bool(enable_safety_filter),
            strict_safety=bool(strict_safety),
        )
        if result.get("error"):
            return None, [], f"❌ {result['error']} - {result.get('reason','')}"
        warning = ""
        if result.get("reason") and not result.get("error"):
            warning = f" ⚠️ {result['reason']}"
        img = PILImage.open(io.BytesIO(base64.b64decode(result["image_base64"])))
        status = "🔄 Cached" if result.get("from_cache") else "✨ Generated"
        info = f"⚡ {result['generation_time']:.2f}s | {status}{warning}"
        return img, [], info

    with gr.Blocks(title="AI Image Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# AI Image Generator")
        with gr.Row():
            with gr.Column(scale=2):
                prompt            = gr.Textbox(label="✨ Prompt", lines=3, placeholder="…")
                negative_prompt   = gr.Textbox(label="🚫 Negative Prompt", lines=2, placeholder="…")
                steps             = gr.Slider(1, 50, 20, label="🔢 Inference Steps")
                guidance          = gr.Slider(1.0, 20.0, 7.5, step=0.5, label="🎯 Guidance Scale")
                width             = gr.Dropdown([512,768,1024], value=512, label="📐 Width")
                height            = gr.Dropdown([512,768,1024], value=512, label="📐 Height")
                seed              = gr.Number(value=-1, label="🎲 Seed (-1 random)", precision=0)
                add_watermark     = gr.Checkbox(value=True, label="🏷️ Add Watermark")
                watermark_text    = gr.Textbox(value="AI Generated", label="✏️ Watermark Text")
                force_regenerate  = gr.Checkbox(value=False, label="♻️ Force Regenerate")
                safety_filter     = gr.Checkbox(value=True, label="🔒 Enable Safety Filter")
                strict_safety     = gr.Checkbox(value=False, label="❗ Strict Safety Mode")
                batch_prompts     = gr.Textbox(
                    label="🗒️ Batch Prompts (one per line)",
                    placeholder="Enter one prompt per line to run batch…",
                    lines=4
                )
                btn               = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=3):
                out_img    = gr.Image(label="Single Result", height=400, show_download_button=True)
                out_gallery = gr.Gallery(label="Batch Results", columns=2, height="auto")
                out_info   = gr.Textbox(label="Info", interactive=False, lines=2)

        btn.click(
            fn=generate_image_ui,
            inputs=[
                prompt,
                negative_prompt,
                steps,
                guidance,
                width,
                height,
                seed,
                add_watermark,
                watermark_text,
                force_regenerate,
                safety_filter,
                strict_safety,
                batch_prompts,
            ],
            outputs=[out_img, out_gallery, out_info],
            show_progress=True,
        )

    mount_gradio_app(app=api, blocks=demo, path="/")
    return api
