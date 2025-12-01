import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import importlib.util
import models.classification_models.VGG as vgg

WEIGHTS_ROOT = "weights"
CLS_SAVE = os.path.join(WEIGHTS_ROOT, "classification_models")
SEG_SAVE = os.path.join(WEIGHTS_ROOT, "segmentation_models")

CLASSES = ["COVID", "Healthy", "Non-COVID"]
IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_class_from_file(file_path, class_name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def add_classification_head(model, num_classes, freeze_features=True, dropout_p=0.5):
    """
    Adapt a pretrained model to a new number of classes.
    Optionally freeze the feature extractor for transfer learning.
    """
    if hasattr(model, 'fc'):
        # ResNet-like
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(num_features, num_classes)
        )
    elif hasattr(model, 'classifier'):
        # VGG-like
        classifier_seq = model.classifier
        if isinstance(classifier_seq, nn.Sequential):
            last_linear_idx = None
            for i in reversed(range(len(classifier_seq))):
                if isinstance(classifier_seq[i], nn.Linear):
                    last_linear_idx = i
                    break
            if last_linear_idx is not None:
                in_features = classifier_seq[last_linear_idx].in_features
                classifier_seq[last_linear_idx] = nn.Linear(in_features, num_classes)
                model.classifier = classifier_seq
        # Freeze features if requested
        if freeze_features:
            for param in model.features.parameters():
                param.requires_grad = False
    return model

def get_classification_model(name):
    name_lower = name.lower()
    # Use pretrained=True to ensure the feature layers (features) are initialized with ImageNet weights
    pretrained_flag = True 
    try:
        if "resnet" in name_lower:
            model = torch.hub.load('pytorch/vision:v0.10.0', name_lower, weights="IMAGENET1K_V1")
        elif "vgg" in name_lower:
            hub_name = name_lower + "_bn"
            model = torch.hub.load('pytorch/vision:v0.10.0', hub_name, weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unknown classification model: {name}")
    except Exception as e:
        raise RuntimeError(f"Failed to load base model architecture for {name}. Ensure PyTorch/torchvision are installed and compatible. Hub Error: {e}")
    
    # Overwrite the classification head to match the number of target classes (3)
    model = add_classification_head(model, len(CLASSES)) 
    return model

class PlaceholderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_conv = nn.Conv2d(3, 1, kernel_size=1) 
    def forward(self, x): 
        return self.dummy_conv(x) 

def get_segmentation_model(name):
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
            "class": "AttentionUNet"
        },
        "r2unet": {
            "file": "R2U_Net.py",
            "class": "R2U_Net"
        },
        "r2attentionnunet": {
            "file": "R2AttU_Net.py",
            "class": "R2AttU_Net"
        }
    }
    if name_lower not in model_configs:
        raise ValueError(f"Unknown segmentation model: {name}")
    config = model_configs[name_lower]
    model_filepath = os.path.join(project_root, "models", "segmentation_models", config["file"])
    try:
        ModelClass = _load_class_from_file(
            file_path=model_filepath,
            class_name=config["class"],
            module_name=name_lower
        )
        return ModelClass()
    except Exception as e:
        print(f"FATAL WARNING: Cannot import segmentation model architecture for {name}.")
        print(f"Attempted to load from: {model_filepath}")
        print(f"Attempted to load class: {config['class']}")
        print(f"Import/Load Error details: {e}")
        print("\n*** ACTION REQUIRED: CHECK FILE NAME AND CLASS NAME CASE ***")
        print(f"Please verify the file name is **{config['file']}** and the class inside is **{config['class']}** in the `models/segmentation_models/` folder.")
        print("***\n")
        return PlaceholderModel()

def preprocess_for_vgg(pil_image):
    """
    Prepares a PIL image for VGG16/VGG19 inference.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # VGG expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return preprocess(pil_image).unsqueeze(0)  # Add batch dim

# [Pipeline Class]

class Pipeline:
    def __init__(self):
        self.classification_model = None
        self.segmentation_model = None
        self.val_transform = self._get_validation_transform()
        os.makedirs(CLS_SAVE, exist_ok=True)
        os.makedirs(SEG_SAVE, exist_ok=True)

    def _get_validation_transform(self):
        return A.Compose([
            A.Resize(height=IMG_SIZE, width=IMG_SIZE), 
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def _load_models(self, classification_name, segmentation_name):
        # [CLASSIFICATION MODEL LOADING]
        if not self.classification_model or self.classification_model.__class__.__name__.lower() != classification_name.lower():
            self.classification_model = None
            success = False
            try:
                # 1. Load the model architecture (initialized with ImageNet weights and adapted head)
                self.classification_model = get_classification_model(classification_name)
                cls_weights_path = os.path.join(CLS_SAVE, f"{classification_name}_best_acc.pt")
                if os.path.exists(cls_weights_path):
                    if 'vgg' in classification_name.lower():
                        state_dict = torch.load(cls_weights_path, map_location=DEVICE)
                        if 'classifier.7.weight' in state_dict and 'classifier.7.bias' in state_dict:
                            w = state_dict['classifier.7.weight']
                            b = state_dict['classifier.7.bias']
                            in_features = self.classification_model.classifier[6].in_features
                            self.classification_model.classifier[6] = nn.Linear(in_features, 3)
                            self.classification_model.classifier[6].weight.data.copy_(w)
                            self.classification_model.classifier[6].bias.data.copy_(b)
                    else:
                        state_dict = torch.load(cls_weights_path, map_location=DEVICE)
                    # strict=False to allow mismatched final layer
                    model_load_result = self.classification_model.load_state_dict(state_dict, strict=False)
                    if model_load_result.missing_keys:
                        print(f"Missing keys (usually last layer): {model_load_result.missing_keys}")
                else:
                    print("Weights file not found. Using pretrained ImageNet weights only.")

                print(f"Successfully loaded Classification Model: {classification_name} (Features and intermediate classifier layers loaded; final layer is custom 3-class).")
                success = True
            
            except Exception as e:
                # 3. Catch FileNotFoundError or general exceptions
                print(f"CRITICAL ERROR: Failed to load classification model {classification_name}. File or general error: {e}")
                self.classification_model = None 

            # 4. Finalize model setup if loading was successful
            if self.classification_model and success:
                self.classification_model.eval()
                self.classification_model.to(DEVICE)
            elif self.classification_model and not success:
                # If architecture loaded but weights failed (e.g., partial load failed), reset model
                 self.classification_model = None


        # [SEGMENTATION MODEL LOADING]
        if not self.segmentation_model or self.segmentation_model.__class__.__name__.lower() != segmentation_name.lower():
            self.segmentation_model = None
            try:
                self.segmentation_model = get_segmentation_model(segmentation_name)
                if not isinstance(self.segmentation_model, PlaceholderModel):
                    seg_weights_path = os.path.join(SEG_SAVE, f"{segmentation_name}_best_loss.pt")
                    
                    print(f"Attempting to load SEG weights from: {seg_weights_path}")
                    # Load state dict with weights_only=True
                    self.segmentation_model.load_state_dict(
                        torch.load(seg_weights_path, map_location=DEVICE, weights_only=True)
                    )
                    self.segmentation_model.eval()
                    self.segmentation_model.to(DEVICE)
                    print(f"Successfully loaded Segmentation Model: {segmentation_name}")
                else:
                    print(f"Using Placeholder for Segmentation Model: {segmentation_name}. Segmentation will be skipped.")
            except Exception as e:
                print(f"CRITICAL ERROR: Failed to load segmentation weights for {segmentation_name}. Check weights path or architecture mismatch. Error: {e}")
                self.segmentation_model = None


    def _predict_classification(self, img_tensor):
        if self.classification_model is None:
            return "FATAL ERROR: Classification Model Not Loaded", 0.0
        with torch.no_grad():
            logits = self.classification_model(img_tensor.to(DEVICE))
            probs = F.softmax(logits, dim=1)[0]
            confidence, predicted_index = torch.max(probs, 0)
            prediction = CLASSES[predicted_index.item()]
            
        return prediction, confidence.item() * 100

    def _predict_segmentation(self, img_tensor):
        if self.segmentation_model is None or isinstance(self.segmentation_model, PlaceholderModel):
            return None 
        with torch.no_grad():
            logits = self.segmentation_model(img_tensor.to(DEVICE))
            mask = torch.sigmoid(logits).cpu().squeeze(0).squeeze(0)
            mask_np = (mask.numpy() > 0.5).astype(np.uint8) * 255 
        return mask_np


    def process_image(self, pil_image, segmentation_model_name, overlay_opacity=0.5):
        if pil_image is None:
            return "No Image Uploaded", 0.0, None, "Please upload an image to begin analysis."
        original_img_np = np.array(pil_image.convert("RGB"))
        H, W, _ = original_img_np.shape
        img_np = np.array(pil_image.convert("RGB")) 
        if self.classification_model.__class__.__name__ in ['VGG16', 'VGG19']:
            img_tensor = preprocess_for_vgg(pil_image)
        else:
            transformed = self.val_transform(image=img_np)
            img_tensor = transformed['image'].unsqueeze(0)

        # Ensure models are loaded before prediction
        cls_name_to_load = self.classification_model.__class__.__name__ if self.classification_model else 'ResNet50'
        self._load_models(cls_name_to_load, segmentation_model_name)
        
        prediction, confidence = self._predict_classification(img_tensor)
        output_img = None
        analysis_text = f"Diagnosis: {prediction}\nConfidence: {confidence:.2f}%\n"

        if "ERROR" in prediction:
            analysis_text = prediction 
        elif prediction != "COVID": # Use explicit check
            analysis_text += "\nRecommendation: Consult a medical professional for final diagnosis. The model suggests no severe COVID-19 pathology."
        else:
            mask_np = self._predict_segmentation(img_tensor)
            if mask_np is not None:
                mask_resized = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
                original_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)
                # Ensure original_bgr is not empty/None before overlay operations
                if original_bgr.size > 0:
                    overlay = np.zeros_like(original_bgr, dtype=np.uint8)
                    overlay[mask_resized == 255] = (0, 0, 255)
                    blended_bgr = cv2.addWeighted(original_bgr, 1.0, overlay, overlay_opacity, 0)
                    output_img = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
                    analysis_text += "\nInfection areas have been highlighted in red (segmentation model)."
                else:
                     analysis_text += "\nWARNING: Image conversion failed. Cannot highlight infection areas."
            else:
                 analysis_text += "\nWARNING: Segmentation model failed to load. Cannot highlight infection areas."
            
        return prediction, confidence, output_img, analysis_text