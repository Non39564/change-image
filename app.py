import streamlit as st
from PIL import Image
import processor

st.set_page_config(page_title="NSFW Inpainting", layout="wide")

st.title("AI Inpainting Tool")

# Sidebar for settings
st.sidebar.header("Settings")
# Default to a smaller model for CPU compatibility. 
# Users can change this to "andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING" if they have powerful hardware.
model_id = st.sidebar.text_input("Model ID", "runwayml/stable-diffusion-inpainting")

# Cache the model loading to prevent reloading on every interaction
@st.cache_resource
def get_model(model_id):
    return processor.load_inpainting_model(model_id)

@st.cache_resource
def get_segmentation_model():
    return processor.load_segmentation_model()

pipe = None
seg_pipeline = None

try:
    with st.spinner("Loading AI Models... This might take a while."):
        pipe = get_model(model_id)
        seg_pipeline = get_segmentation_model()
    st.sidebar.success("All Models Loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading models: {e}")

# Main interface
col1, col2 = st.columns(2)

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("Input Image")
        st.image(input_image, use_container_width=True)

    # Generate mask automatically
    with st.spinner("Generating clothing mask..."):
        mask_image = processor.create_clothing_mask(input_image, seg_pipeline)
    
    with col2:
        st.subheader("Generated Mask")
        st.image(mask_image, use_container_width=True)
        st.caption("White area will be regenerated")

    # Prompt inputs
    st.subheader("Generation Details")
    default_prompt = (
        "explicit erotic nude woman, full naked body, "
        "exposed breasts, visible nipples, show pussy, "
        "realistic female anatomy, "
        "soft erotic lighting, NSFW"
    )
    prompt = st.text_area("Prompt", default_prompt, height=100)
    
    default_negative = "safe, censored, underwear, swimsuit, bra, panties, covered body"
    negative_prompt = st.text_input("Negative Prompt", default_negative)

    # Advanced parameters
    with st.expander("Advanced Parameters"):
        strength = st.slider("Strength", 0.0, 1.0, 0.99, 0.01, help="Higher strength means more replacement of the original image area.")
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 8.5, 0.1)
        # denoising_start is SDXL specific and not used effectively in standard pipeline fallback
        # denoising_start = st.slider("Denoising Start", 0.0, 1.0, 0.2, 0.01) 
        num_inference_steps = st.slider("Steps", 10, 100, 35, 1)

    if st.button("Generate", type="primary"):
        # Progress bar
        progress_bar = st.progress(0)
        
        def update_progress(progress):
            progress_bar.progress(progress)

        with st.spinner("Generating..."):
            try:
                result = processor.generate_image(
                    pipe,
                    input_image,
                    mask_image,
                    prompt,
                    negative_prompt,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    callback=update_progress
                )
                
                # Clear progress bar on completion
                progress_bar.empty()
                
                st.subheader("Result")
                st.image(result, use_container_width=True)
                
                # Download button
                import io
                buf = io.BytesIO()
                result.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Result",
                    data=byte_im,
                    file_name="result.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error generating image: {e}")

else:
    st.info("Please upload an image to start.")
