import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from diffusers import StableDiffusionXLInpaintPipeline
import gc
import os

def load_segmentation_model():
    """
    Loads the Segformer model for clothing segmentation.
    Should be cached by Streamlit.
    """
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    return processor, model

def create_clothing_mask(img_pil, segmentation_pipeline=None):
    """
    Creates a mask specifically for clothing items using Segformer.
    Labels: 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 8: Belt, 17: Scarf
    """
    # Load model if not provided (slower)
    if segmentation_pipeline is None:
        processor, model = load_segmentation_model()
    else:
        processor, model = segmentation_pipeline

    # Process image
    inputs = processor(images=img_pil, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Upsample logits to original image size
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=img_pil.size[::-1], # (height, width)
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    
    # Define clothing labels to include in mask
    # 4: Upper-clothes, 5: Skirt, 6: Pants, 7: Dress, 8: Belt, 16: Bag, 17: Scarf
    clothing_labels = [4, 5, 6, 7, 8, 16, 17]
    
    mask = np.zeros_like(pred_seg, dtype=np.uint8)
    for label in clothing_labels:
        mask[pred_seg == label] = 255
        
    # Optional: Dilate slightly to cover edges
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask_img = Image.fromarray(mask, mode="L")
    return mask_img

def load_inpainting_model(model_id="runwayml/stable-diffusion-inpainting"):
    """
    Loads the Inpainting pipeline.
    Should be cached by Streamlit.
    """
    # Memory cleanup
    gc.collect()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    if device == "cuda":
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Load pipeline
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline

    # Determine which pipeline class to use based on model ID
    if "sxed" in model_id.lower() or "xl" in model_id.lower():
        PipelineClass = StableDiffusionXLInpaintPipeline
    else:
        PipelineClass = StableDiffusionInpaintPipeline

    try:
        pipe = PipelineClass.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        # Fallback 1: Try standard load with safetensors (sometimes low_cpu_mem_usage fails specifically)
        try:
            print(f"Error loading with optimization: {e}. Trying standard load with safetensors.")
            pipe = PipelineClass.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=True
            )
        except Exception as e2:
            # Fallback 2: Try without safetensors (some older models only have .bin for fp32)
            print(f"Error loading with safetensors: {e2}. Trying without safetensors.")
            pipe = PipelineClass.from_pretrained(
                model_id,
                torch_dtype=dtype,
                use_safetensors=False
            )

    # Optimization
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")
    else:
        # On CPU, ensure it is on CPU
        pipe.to("cpu")
        # Optimization for CPU execution (saves memory)
        pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")

    return pipe

def generate_image(pipe, init_image, mask_image, prompt, negative_prompt, 
                   strength=0.35, guidance_scale=8.5, denoising_start=0.2, num_inference_steps=35):
    """
    Generates the inpainted image.
    """
    # Memory optimization
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get pipeline type to decide resolution
    is_sdxl = isinstance(pipe, StableDiffusionXLInpaintPipeline)
    target_size = (1024, 1024) if is_sdxl else (512, 512)

    # Resize images to target size for processing
    # Save original size to resize back later if needed
    original_size = init_image.size
    
    init_image = init_image.resize(target_size, Image.LANCZOS).convert("RGB")
    mask_image = mask_image.resize(target_size, Image.NEAREST).convert("L")

    # Lower inference steps for CPU execution to prevent timeouts
    if pipe.device.type == "cpu":
        num_inference_steps = min(num_inference_steps, 20)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    
    # Optional: Resize back to original size? 
    # For now, let's keep it at generation size to match quality expectation
    # result = result.resize(original_size, Image.LANCZOS)
    
    return result
