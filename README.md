# Diffusion Model on Modal

An AI image generation service built with **Modal**, **Stable Diffusion 2.1**, and **Gradio**. Features intelligent caching, automatic watermarking, content safety filtering, and both single/batch generation capabilities.

## 🚀 Features

- **High-Performance Generation**: Stable Diffusion 2.1 with GPU acceleration (A10G)
- **Intelligent Caching**: Automatic result caching based on generation parameters
- **Automatic Watermarking**: Configurable watermarks with position, opacity, and text options
- **Content Safety**: Built-in NSFW and toxicity detection with configurable strictness
- **Batch Processing**: Generate multiple images from a list of prompts
- **Web Interface**: Clean Gradio UI for easy interaction
- **Memory Optimization**: XFormers integration and attention slicing for efficient GPU usage
- **Persistent Storage**: Modal volumes for model and image caching

## 🛠 Tech Stack

- **Compute Platform**: [Modal](https://modal.com/) - Serverless GPU infrastructure
- **ML Framework**: PyTorch, Diffusers, Transformers
- **Model**: Stable Diffusion 2.1 by Stability AI
- **Safety**: Detoxify (toxicity detection), Better Profanity
- **UI**: Gradio web interface
- **Image Processing**: Pillow (PIL)
- **Optimization**: XFormers for memory-efficient attention

## 📋 Prerequisites

- Modal account and CLI setup
- Python 3.10+
- GPU quota on Modal (A10G recommended)

## 🏃‍♂️ Quick Start

### 1. Install Modal CLI

```bash
pip install modal
modal setup
```

### 2. Run The application

```modal serve model.py```

After deployment, Modal will provide a URL for your Gradio interface:

```
✓ Initialized. View run at https://modal.com/apps/<name>/main/ap-hmEks9TEElWXoiKBrqH8sV
✓ Created objects.
├── 🔨 Created mount model.py
├── 🔨 Created function generate_image_cached.
├── 🔨 Created function clear_cache.
├── 🔨 Created function get_cache_stats.
├── 🔨 Created function multiple_prompts_cached.
└── 🔨 Created web function gradio_app => https://<name>--stable-diffusion-2-1-cached-watermarked-5f0e44-dev.modal.run (label truncated)
```

## 🎨 Web Interface Guide

### Single Image Generation
1. Enter your prompt in the "✨ Prompt" field
2. Optionally add negative prompts to avoid unwanted elements
3. Adjust generation parameters (steps, guidance scale, dimensions)
4. Configure watermark settings
5. Click "Generate"

### Batch Generation
1. Leave the single prompt field empty
2. Enter multiple prompts in "🗒️ Batch Prompts" (one per line)
3. Configure shared parameters
4. Click "Generate" to process all prompts

### Parameter Guide

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| **Inference Steps** | Quality vs speed tradeoff | 15-30 (20 default) |
| **Guidance Scale** | Prompt adherence strength | 5.0-15.0 (7.5 default) |
| **Width/Height** | Output dimensions | 512, 768, or 1024px |
| **Seed** | Reproducibility control | -1 for random |

## 🔒 Safety Features

### Content Filtering
- **Keyword Detection**: Blocks common NSFW/violent terms
- **AI-based Detection**: Detoxify model for toxicity scoring
- **Configurable Strictness**: 
  - Normal: Adds negative prompts for flagged content
  - Strict: Completely blocks generation

### Safety Thresholds
```python
thresholds = {
    'toxicity': 0.7,
    'severe_toxicity': 0.5,
    'obscene': 0.7,
    'threat': 0.7,
    'insult': 0.8,
    'identity_attack': 0.7
}
```

## 💾 Caching System

### How It Works
- **Cache Key**: SHA256 hash of all generation parameters
- **Storage**: Persistent Modal volumes with pickle serialization
- **Metadata**: JSON metadata for easy inspection
- **Automatic**: No manual cache management needed

### Cache Invalidation
Cache is automatically bypassed when:
- `force_regenerate=True` is set
- Any generation parameter changes
- Watermark settings change

## 📊 Performance & Scaling

### Resource Configuration
- **GPU**: A10G (16GB VRAM)
- **Memory**: 16GB RAM
- **Timeout**: 5 minutes per generation
- **Auto-scaling**: Scales to zero when idle
- **Concurrent**: Up to 100 concurrent requests

### Optimization Features
- XFormers memory-efficient attention
- Attention slicing for large images
- Model CPU offloading as fallback
- Persistent model caching

## 🔧 Configuration Options

### Watermark Customization

```python
watermark_settings = {
    "text": "Custom Watermark",
    "position": "bottom-right",  # top-left, top-right, bottom-left, bottom-right, center
    "opacity": 0.7,              # 0.0 to 1.0
    "font_size": None,           # Auto-calculated if None
    "font_color": (255, 255, 255), # RGB tuple
    "margin": 10                 # Pixels from edge
}
```

### Safety Configuration

```python
safety_config = {
    "enable_safety_filter": True,
    "strict_safety": False,      # True = block, False = add negative prompts
    "custom_keywords": [...],    # Additional blocked keywords
    "toxicity_threshold": 0.7    # Adjust sensitivity
}
```

## 📈 Monitoring

### Performance Metrics
- Generation time per image
- Cache hit/miss ratios  
- GPU utilization
- Memory usage patterns