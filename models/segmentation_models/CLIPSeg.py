import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from PIL import Image
import numpy as np

# Default text prompt for lung segmentation
DEFAULT_TEXT_PROMPT = "lungs"


class CLIPSegForSegmentation(nn.Module):
    def __init__(
        self,
        model_name="CIDAS/clipseg-rd64-refined",
        text_prompt="lungs",
        device=None,
    ):
        super(CLIPSegForSegmentation, self).__init__()

        # Load CLIPSeg model and processor
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            model_name
        ).to(self.device)
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.text_prompt = text_prompt

    def forward(self, pixel_values, input_ids, attention_mask):
        # Get segmentation logits
        outputs = self.clipseg_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits

    def preprocess_image(self, image, text_prompt=None):
        if text_prompt is None:
            text_prompt = self.text_prompt

        inputs = self.processor(
            text=[text_prompt], images=image, return_tensors="pt", padding=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def predict(self, image, text_prompt=None, threshold=0.5, return_probs=False):
        self.eval()
        with torch.no_grad():
            # Handle different image formats
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype("uint8"), "RGB")

            # Preprocess
            inputs = self.preprocess_image(image, text_prompt)
            
            # Forward pass
            logits = self.forward(
                inputs["pixel_values"],
                inputs["input_ids"],
                inputs["attention_mask"],
            )
            
            # Get probabilities
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            # Resize to original image size
            original_size = image.size[::-1]  # (H, W)
            if probs.shape != original_size:
                probs_tensor = torch.from_numpy(probs).unsqueeze(0).unsqueeze(0)
                probs = (
                    F.interpolate(
                        probs_tensor,
                        size=original_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze()
                    .numpy()
                )

            # Create binary mask
            binary_mask = (probs > threshold).astype(np.uint8)

            if return_probs:
                return binary_mask, probs
            return binary_mask

    def freeze_encoder(self):
        """Freeze encoder parameters (only train decoder)."""
        for name, param in self.clipseg_model.named_parameters():
            if "decoder" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def freeze_decoder(self):
        """Freeze decoder parameters."""
        for name, param in self.clipseg_model.named_parameters():
            if "decoder" in name:
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.clipseg_model.parameters():
            param.requires_grad = True

    def get_trainable_params(self, decoder_only=True):
        if decoder_only:
            return filter(
                lambda p: p.requires_grad and any(
                    name.find("decoder") != -1
                    for name, param in self.clipseg_model.named_parameters()
                    if param is p
                ),
                self.clipseg_model.parameters(),
            )
        else:
            return filter(lambda p: p.requires_grad, self.parameters())


def create_clipseg_model(
    text_prompt=None,
    device="cuda",
    model_name="CIDAS/clipseg-rd64-refined",
):
    if text_prompt is None:
        text_prompt = DEFAULT_TEXT_PROMPT

    model = CLIPSegForSegmentation(
        model_name=model_name,
        text_prompt=text_prompt,
        device=device,
    )

    return model


def load_clipseg_model(
    checkpoint_path,
    text_prompt=None,
    device="cuda",
    model_name="CIDAS/clipseg-rd64-refined",
):
    model = create_clipseg_model(text_prompt, device, model_name)
    model.clipseg_model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    model.to(device)

    return model
