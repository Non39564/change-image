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

def load_inpainting_model(model_id="andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING"):
    """
    Loads the Stable Diffusion XL Inpainting pipeline.
    Should be cached by Streamlit.
    """
    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float16 for GPU, float32 for CPU (MPS/Mac could also use float16 usually, but let's be safe)
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True
    )

    # Optimization
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        pipe.enable_attention_slicing("max")
    else:
        # On CPU, just move the pipeline to CPU explicitly or leave it
        # Generating on CPU will be slow
        pipe.to("cpu")
    
    return pipe

def generate_image(pipe, init_image, mask_image, prompt, negative_prompt, 
                   strength=0.35, guidance_scale=8.5, denoising_start=0.2, num_inference_steps=35):
    """
    Generates the inpainted image.
    """
    # Ensure images are resized to 1024x1024 as in original script (SDXL prefers 1024x1024)
    init_image = init_image.convert("RGB").resize((1024,1024))
    mask_image = mask_image.convert("L").resize((1024,1024))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        strength=strength,
        guidance_scale=guidance_scale,
        denoising_start=denoising_start,
        num_inference_steps=num_inference_steps
    ).images[0]
    
    return result
