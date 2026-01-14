import gradio as gr
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import cv2

# Load the multimodal model (detects + generates explanations)
model_id = "YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map="auto"  # Auto uses GPU if available
)
processor = AutoProcessor.from_pretrained(model_id)

def predict_disease(image):
    if image is None:
        return None, "No image provided."
    
    # Prompt the model to detect disease and provide detailed info
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this plant leaf image carefully. What crop is it? What disease (if any) does it show? Describe the visible symptoms in detail. What are the likely causes? How can it be treated or prevented? Provide confidence level if possible."},
                {"type": "image"},
            ],
        }
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate response
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract the assistant's response (cleaner output)
    assistant_response = generated_text.split("assistant")[1].strip() if "assistant" in generated_text else generated_text
    
    output_text = f"""
**Model-Generated Analysis:**

{assistant_response}

*(All information above is generated directly by the multimodal model based on the image — no hardcoded data used.)*
    """
    
    return image, output_text

# Webcam capture
def webcam_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, "Could not open webcam."
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None, "Failed to capture image."
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    return pil_image

# Gradio interface
with gr.Blocks(title="Advanced Crop Disease Detection (Multimodal AI)") as demo:
    gr.Markdown("# 🌿 Crop Disease Detection with AI-Generated Explanations")
    gr.Markdown("Upload or capture a leaf image. The AI model analyzes it and generates disease info, symptoms, causes, and treatment advice directly.")
    
    with gr.Row():
        with gr.Column():
            upload_input = gr.Image(label="Upload Leaf Image", type="pil")
            webcam_btn = gr.Button("Capture from Webcam")
            webcam_output = gr.Image(label="Captured Image", type="pil")
        
        with gr.Column():
            result_image = gr.Image(label="Input Image", type="pil")
            result_text = gr.Textbox(label="AI-Generated Diagnosis & Advice", lines=15)
    
    upload_btn = gr.Button("Analyze Uploaded Image")
    analyze_webcam_btn = gr.Button("Analyze Captured Webcam Image")
    
    upload_btn.click(predict_disease, inputs=upload_input, outputs=[result_image, result_text])
    webcam_btn.click(webcam_capture, outputs=[webcam_output, result_text])
    analyze_webcam_btn.click(predict_disease, inputs=webcam_output, outputs=[result_image, result_text])
demo.launch()