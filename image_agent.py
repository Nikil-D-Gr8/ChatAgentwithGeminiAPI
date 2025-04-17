# ⬅️ Import modules
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import base64
import io
from pathlib import Path
from typing import Union

class PropertyIssueDetectionAgent:
    def __init__(self):
        print("Initializing Property Issue Detection Agent...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        print("Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        print("Property Issue Detection Agent initialized successfully")

    def analyze_image(self, image_data: str, user_query: str) -> dict:
        """Analyze image and return caption and detected issues"""
        try:
            # Decode base64 image
            if isinstance(image_data, str):
                if "base64," in image_data:
                    image_data = image_data.split("base64,")[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(image_data, (str, Path)):
                image = Image.open(image_data)
            else:
                raise ValueError("Unsupported image format")

            # Process with BLIP
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract potential issues from the caption
            detected_issues = []
            if 'mold' in caption.lower():
                detected_issues.append({
                    'issue': 'Mold detected',
                    'severity': 'High',
                    'description': 'Presence of mold indicates potential health hazard and moisture problems'
                })
            if 'damage' in caption.lower():
                detected_issues.append({
                    'issue': 'Structural damage',
                    'severity': 'High',
                    'description': 'Visible damage that may require immediate attention'
                })
            if 'crack' in caption.lower():
                detected_issues.append({
                    'issue': 'Cracks present',
                    'severity': 'High',
                    'description': 'Cracks may indicate structural issues or settling'
                })
            
            return {
                "description": caption,
                "detected_issues": detected_issues
            }

        except Exception as e:
            print(f"Error processing image: {e}")
            raise

if __name__ == "__main__":
    agent = PropertyIssueDetectionAgent()
    result = agent.analyze_image("moldup.jpeg")

    print("\n--- Results ---")
    print("Description:", result["description"])
    print("Detected issues:", result["detected_issues"])




