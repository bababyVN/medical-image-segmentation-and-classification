import gradio as gr
import torch.nn.functional as F
import albumentations as A
# Safely import only the required class
from design.pipeline import Pipeline 
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Utility to safely load CSS
def get_css(css_path):
    """Safely loads CSS or returns empty string if the file is missing."""
    try:
        with open(css_path, 'r') as f:
            custom = f.read()
        return custom
    except FileNotFoundError:
        print(f"WARNING: CSS file not found at {css_path}. Continuing without custom styling.")
        return "" 

def create_interface():
    custom = get_css('design/design.css') 
    processor = Pipeline()

    with gr.Blocks(css=custom, theme=gr.themes.Soft(primary_hue='teal', secondary_hue='blue')) as interface:
        with gr.Column(variant="compact"):
            gr.Markdown("# Lungs Radiography Analysis", elem_classes='heading')
            gr.Markdown("""
                Upload/ Drop a chest X-ray image for COVID-19 diagnosis and analysis. 
            """)
        
        with gr.Row(equal_height=True):
            # [MODEL SELECTION]
            # Changed scale from 0.2 to 2 to prevent Gradio UserWarning
            with gr.Column(scale=2): 
                classification_dropdown = gr.Dropdown(
                    choices=['ResNet50', 'ResNet18', 'VGG16', 'VGG19'],
                    value='ResNet50',
                    label='Classification Model',
                )
                segmentation_dropdown = gr.Dropdown(
                    choices=['ResNetUnet', 'AttentionUNet', 'R2Unet', 'R2AttUnet'],
                    value='ResNetUnet',
                    label='Segmentation Model'
                )
                overlay_opacity = gr.Slider(
                    minimum=0.1, maximum=1.0, step=0.05, value=0.5,
                    label="Overlay Opacity (for COVID mask)",
                    interactive=True
                )

            # [UPLOAD IMAGE SECTION]
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Chest X-ray",
                    height=400,
                    elem_classes="upload-image",
                    # *** FIX 1: Ensure input is a PIL Image object for .convert() method ***
                    type="pil" 
                )

                # [BUTTON]
                with gr.Row():
                    submit_btn = gr.Button("Analyze Image", variant="primary", elem_classes='primary-button', scale=2)
                    clear_btn = gr.Button('Clear', variant='secondary', scale=1)
                    
            with gr.Column():
                with gr.Group(elem_classes='results-container'):                    
                    output_image = gr.Image(
                        label="Infection Areas (COVID)",
                        visible=False,
                        height=400
                    )

                with gr.Row(equal_height=True):
                    diagnosis_label = gr.Label(label="Diagnosis Conclusion", elem_classes='results-container')
                    confidence_label = gr.Label(label="Confidence Score", elem_classes='results-container')
                
                with gr.Row():
                    diagnosis_text = gr.Textbox(
                                label="Diagnosis Details",
                                visible=False,
                                container=False,
                                lines=5
                            )
        
        # [HELP SECTION]    
        with gr.Accordion("Information", open=False):
                    gr.Markdown("""
                ### Tutorial
                1. Click the upload button/ Drag and drop a chest X-ray image.
                2. Select your trained Classification and Segmentation models.
                3. Choose 'Analyze Image'.
                4. Review the results: For COVID cases, the segmentation mask is overlaid on the image in red.
            """)
                    
        def clear_inputs():
            # **FIXED**: Using POSITIONAL RETURN for stability
            return (
                None, # diagnosis_label
                None, # confidence_label
                gr.update(value=None, visible=False), # output_image
                gr.update(value="", visible=False) # diagnosis_text
            )
        
        def handle_prediction(image, classification_model, segmentation_model, opacity):
            # 1. Load weights and initialize models
            processor._load_models(classification_model, segmentation_model)            
            
            # 2. Process image and get results
            prediction, confidence, output_img, analysis_text = processor.process_image(
                image, segmentation_model, overlay_opacity=opacity
            )
            
            # Determine CSS class for confidence label
            confidence_class = (
                "confidence-high" if confidence > 90
                else "confidence-medium" if confidence > 70
                else "confidence-low"
            )
            
            is_covid_prediction = prediction == "COVID" and output_img is not None
            
            # 3. Update the UI - POSITIONAL RETURN (MUST MATCH outputs=[...] list)
            return (
                # 1. diagnosis_label
                prediction, 

                # 2. confidence_label
                gr.update(
                    value=f"Confidence: {confidence:.2f}%",
                    elem_classes=[confidence_class]
                ),

                # 3. output_image
                gr.update(value=output_img, visible=is_covid_prediction),

                # 4. diagnosis_text
                gr.update(value=analysis_text, visible=True)
            )

        # Event Listeners
        submit_btn.click(
            fn=handle_prediction,
            inputs=[input_image, classification_dropdown, segmentation_dropdown, overlay_opacity],
            outputs=[diagnosis_label, confidence_label, output_image, diagnosis_text]
        )
        
        clear_btn.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[diagnosis_label, confidence_label, output_image, diagnosis_text]
        )
        
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)