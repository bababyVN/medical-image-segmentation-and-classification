import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import importlib.util # <--- NEW im

# --- Configuration ---
WEIGHTS_ROOT = "weights"
CLS_SAVE = os.path.join(WEIGHTS_ROOT, "classification_models")
SEG_SAVE = os.path.join(WEIGHTS_ROOT, "segmentation_models")

CLASSES = ["COVID", "Healthy", "Non-COVID"]
IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utility Function for Dynamic Loading ---

def _load_class_from_file(file_path, class_name, module_name):
    """Dynamically loads a class from a specific file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    
    # Execute the module code
    spec.loader.exec_module(module)
    
    # Return the class object from the loaded module
    return getattr(module, class_name)

# --- Model Architectures & Definitions (Rest of the file remains similar) ---

def add_classification_head(model, num_classes, dropout_p=0.5):
    """
    Replaces the final classification layer with Dropout and Linear layer,
    matching the structure commonly used in fine-tuning.
    """
    if hasattr(model, 'fc'): # For ResNet family
        num_features = model.fc.in_features
        # Ensure your custom head matches this if you changed it during training
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_features, num_classes)
        )
    elif hasattr(model, 'classifier'): # For VGG family
        # Assuming VGG was trained with a similar classifier replacement
        if len(model.classifier) > 0 and hasattr(model.classifier[-1], 'in_features'):
             in_features = model.classifier[-1].in_features
        elif hasattr(model.classifier, 'in_features'):
             in_features = model.classifier.in_features
        else:
            # Fallback for VGG if structure is unexpected
            print("WARNING: Could not automatically detect VGG classifier input features. Using a default assumption.")
            in_features = 4096 # Common VGG final layer input size
        
        # Replacing the whole classifier with a typical fine-tuning head
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512), # Reduced hidden layer size for safety/flexibility
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes)
        )
    return model

def get_classification_model(name):
    """Loads the base classification model from PyTorch hub and configures the head."""
    name_lower = name.lower()
    
    # Use pretrained=False for older torch versions
    try:
        if "resnet" in name_lower:
            model = torch.hub.load('pytorch/vision:v0.10.0', name_lower, pretrained=False)
        elif "vgg" in name_lower:
            hub_name = name_lower + "_bn"
            model = torch.hub.load('pytorch/vision:v0.10.0', hub_name, pretrained=False)
        else:
            raise ValueError(f"Unknown classification model: {name}")
    except Exception as e:
        raise RuntimeError(f"Failed to load base model architecture for {name}. Ensure PyTorch/torchvision are installed and compatible. Hub Error: {e}")
    
    model = add_classification_head(model, len(CLASSES))
    return model

# Placeholder for segmentation models if architecture files are not present
class PlaceholderModel(nn.Module):
     """Used to prevent crashes if custom model files are not importable."""
     def __init__(self):
         super().__init__()
         # Define a dummy layer matching the expected input/output (3 channels in, 1 channel out)
         self.dummy_conv = nn.Conv2d(3, 1, kernel_size=1) 
     def forward(self, x): 
         # Returns a single channel (mask) tensor of the same spatial size
         return self.dummy_conv(x) 

def get_segmentation_model(name):
    """Loads the segmentation model (requires local model architecture files)."""
    name_lower = name.lower()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..')
    
    
    model_configs = {
        "resnetunet": {
            "file": "ResnetUnet.py",
            "class": "ResNetUnet"
        },
        "attentionunet": {
            "file": "AttentionUNet.py",
            "class": "AttentionUNET"
        },
        "r2unet": {
            "file": "R2U_Net.py",
            "class": "R2U_Net"
        },
        "r2attentionunet": {
            "file": "R2AttU_Net.py",
            "class": "R2AttU_Net"
        }
    }
    
    if name_lower not in model_configs:
        raise ValueError(f"Unknown segmentation model: {name}")
    
    config = model_configs[name_lower]
    model_filepath = os.path.join(project_root, "models", "segmentation_models", config["file"])
    
    try:
        # Load the class dynamically using the file path
        ModelClass = _load_class_from_file(
            file_path=model_filepath,
            class_name=config["class"],
            module_name=name_lower # Use a unique module name
        )
        return ModelClass()
    except Exception as e:
        print(f"FATAL WARNING: Cannot import segmentation model architecture for {name}.")
        print(f"Attempted to load from: {model_filepath}")
        print(f"Attempted to load class: {config['class']}")
        print(f"Import/Load Error details: {e}")
        # Detailed instruction for the user
        print("\n*** ACTION REQUIRED: CHECK FILE NAME AND CLASS NAME CASE ***")
        print(f"Please verify the file name is **{config['file']}** and the class inside is **{config['class']}** in the `models/segmentation_models/` folder.")
        print("***\n")
        return PlaceholderModel()


# --- Pipeline Class ---

class Pipeline:
    def __init__(self):
        self.classification_model = None
        self.segmentation_model = None
        self.val_transform = self._get_validation_transform()

        # Create dummy directories if they don't exist
        os.makedirs(CLS_SAVE, exist_ok=True)
        os.makedirs(SEG_SAVE, exist_ok=True)

    def _get_validation_transform(self):
        """Replicates the validation/inference transforms used during training."""
        return A.Compose([
            A.Resize(height=IMG_SIZE, width=IMG_SIZE), 
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def _load_models(self, classification_name, segmentation_name):
        """Initializes and loads weights for the selected models if not already loaded."""
        
        # 1. CLASSIFICATION MODEL LOADING
        if not self.classification_model or self.classification_model.__class__.__name__.lower() != classification_name.lower():
            self.classification_model = None
            try:
                self.classification_model = get_classification_model(classification_name)
                cls_weights_path = os.path.join(CLS_SAVE, f"{classification_name}_best_acc.pt") 
                
                print(f"Attempting to load CLS weights from: {cls_weights_path}")
                self.classification_model.load_state_dict(torch.load(cls_weights_path, map_location=DEVICE))
                self.classification_model.eval()
                self.classification_model.to(DEVICE)
                print(f"Successfully loaded Classification Model: {classification_name}")
            except Exception as e:
                print(f"CRITICAL ERROR: Failed to load classification model {classification_name}. Check weights path or architecture mismatch. Error: {e}")
                self.classification_model = None 

        # 2. SEGMENTATION MODEL LOADING
        if not self.segmentation_model or self.segmentation_model.__class__.__name__.lower() != segmentation_name.lower():
            self.segmentation_model = None
            try:
                self.segmentation_model = get_segmentation_model(segmentation_name)
                # Only try to load weights if a real model (not the Placeholder) was returned
                if not isinstance(self.segmentation_model, PlaceholderModel):
                    seg_weights_path = os.path.join(SEG_SAVE, f"{segmentation_name}_best_loss.pt")
                    
                    print(f"Attempting to load SEG weights from: {seg_weights_path}")
                    self.segmentation_model.load_state_dict(torch.load(seg_weights_path, map_location=DEVICE))
                    self.segmentation_model.eval()
                    self.segmentation_model.to(DEVICE)
                    print(f"Successfully loaded Segmentation Model: {segmentation_name}")
                else:
                    # Keep the placeholder model loaded
                    print(f"Using Placeholder for Segmentation Model: {segmentation_name}. Segmentation will be skipped.")
            except Exception as e:
                # This will catch errors during weight loading, not architecture loading (which is caught inside get_segmentation_model)
                print(f"CRITICAL ERROR: Failed to load segmentation weights for {segmentation_name}. Check weights path or architecture mismatch. Error: {e}")
                self.segmentation_model = None


    def _predict_classification(self, img_tensor):
        """Performs classification inference."""
        if self.classification_model is None:
            # Explicitly return an error if the model failed to load
            return "FATAL ERROR: Classification Model Not Loaded", 0.0

        with torch.no_grad():
            # Ensure the tensor is moved to the device where the model resides
            logits = self.classification_model(img_tensor.to(DEVICE))
            probs = F.softmax(logits, dim=1)[0]
            
            confidence, predicted_index = torch.max(probs, 0)
            prediction = CLASSES[predicted_index.item()]
            
        return prediction, confidence.item() * 100

    def _predict_segmentation(self, img_tensor):
        """Performs segmentation inference."""
        # Check if the model is the placeholder fallback or None
        if self.segmentation_model is None or isinstance(self.segmentation_model, PlaceholderModel):
            return None 

        with torch.no_grad():
            logits = self.segmentation_model(img_tensor.to(DEVICE))
            mask = torch.sigmoid(logits).cpu().squeeze(0).squeeze(0)
            
            # Threshold the mask
            mask_np = (mask.numpy() > 0.5).astype(np.uint8) * 255 
            
        return mask_np


    def process_image(self, pil_image, segmentation_model_name, overlay_opacity=0.5):
        """Main method to process the image and get results."""
        if pil_image is None:
            return "No Image Uploaded", 0.0, None, "Please upload an image to begin analysis."
        
        # Original size and conversion for overlay
        original_img_np = np.array(pil_image.convert("RGB"))
        H, W, _ = original_img_np.shape
        
        # 1. Preprocess
        # Since pil_image is now a PIL object, we convert it to NumPy here for Albumentations
        img_np = np.array(pil_image.convert("RGB")) 
        transformed = self.val_transform(image=img_np)
        img_tensor = transformed['image'].unsqueeze(0)

        # 2. Classification
        prediction, confidence = self._predict_classification(img_tensor)
        
        output_img = None
        analysis_text = f"Diagnosis: {prediction}\nConfidence: {confidence:.2f}%\n"

        if "ERROR" in prediction:
            analysis_text = prediction # Show the detailed error from _predict_classification
            
        elif prediction == "COVID":
            # 3. Segmentation (Only run if classified as COVID)
            mask_np = self._predict_segmentation(img_tensor)
            
            if mask_np is not None:
                # Resize mask back to the original image dimensions
                mask_resized = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
                
                # Overlay logic
                original_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
                overlay = np.zeros_like(original_bgr, dtype=np.uint8)
                overlay[mask_resized == 255] = (0, 0, 255) # Red in BGR

                blended_bgr = cv2.addWeighted(original_bgr, 1.0, overlay, overlay_opacity, 0)
                output_img = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
                
                analysis_text += "\nInfection areas have been highlighted in red (segmentation model)."
            else:
                 analysis_text += "\nWARNING: Segmentation model failed to load. Cannot highlight infection areas."

        else:
            analysis_text += "\nRecommendation: Consult a medical professional for final diagnosis. The model suggests no severe COVID-19 pathology."
            
        return prediction, confidence, output_img, analysis_text