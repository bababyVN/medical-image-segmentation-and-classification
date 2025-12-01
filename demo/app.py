import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from utils.pipeline import Pipeline, DEVICE
import torchvision.transforms as T

vgg_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_css(css_path):
    with open(css_path, 'r') as f:
        css = f.read()
    return css

def create_interface():
    custom = get_css('demo/design.css') 
    processor = Pipeline()

    with gr.Blocks(css=custom, theme=gr.themes.Soft(primary_hue='teal', secondary_hue='blue')) as interface:
        with gr.Column(variant="compact"):
            gr.Markdown("# Lungs Radiography Analysis", elem_classes='heading')
            gr.Markdown("""
                Upload/ Drop a chest X-ray image for COVID-19 diagnosis and analysis. 
            """)
        
        with gr.Row(equal_height=True):
            # [MODEL SELECTION]
            with gr.Column(scale=0.2): 
                classification_dropdown = gr.Dropdown(
                    choices=['ResNet18', 'ResNet50', 'VGG16', 'VGG19'],
                    value='ResNet18',
                    label='Classification Model',
                )
                segmentation_dropdown = gr.Dropdown(
                    choices=['ResNetUnet', 'AttentionUNet', 'R2Unet', 'R2AttUnet'],
                    value='ResNetUnet',
                    label='Segmentation Model'
                )
                overlay_opacity = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                    label="Overlay Opacity (for COVID mask)",
                    interactive=True
                )

            # [UPLOAD IMAGE SECTION]
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Chest X-ray",
                    height=400,
                    elem_classes="upload-image",
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
            return (None, None, gr.update(value=None, visible=False), gr.update(value="", visible=False))
        
        def handle_prediction(image, classification_model, segmentation_model, opacity):
            if classification_model in ['VGG16', 'VGG19']:
                img_tensor = vgg_transform(image).unsqueeze(0).to(DEVICE)
                prediction, confidence = processor._predict_classification(img_tensor)
            processor._load_models(classification_model, segmentation_model)            
            prediction, confidence, output_img, analysis_text = processor.process_image(
            image, segmentation_model, overlay_opacity=opacity)
    
            confidence_class = ("confidence-high" if confidence > 90 else "confidence-medium" if confidence > 70 else "confidence-low")
            is_covid_prediction = prediction == "COVID" and output_img is not None
    
            return (
                prediction, 
                gr.update(value=f"Confidence: {confidence:.2f}%", elem_classes=[confidence_class]),
                gr.update(value=output_img, visible=is_covid_prediction),
                gr.update(value=analysis_text, visible=True))

        # [Event Listeners]
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