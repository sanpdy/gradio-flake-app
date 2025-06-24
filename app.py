# app.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from ultralytics import YOLO
from transformers import ResNetModel

class FlakeLayerClassifier(nn.Module):
    def __init__(self, num_materials, material_dim, num_classes=4, dropout_prob=0.1, freeze_cnn=False):
        super().__init__()
        self.cnn = ResNetModel.from_pretrained("microsoft/resnet-18")
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        img_feat_dim = self.cnn.config.hidden_sizes[-1]
        self.material_embedding = nn.Embedding(num_materials, material_dim)
        self.dropout = nn.Dropout(dropout_prob)

        self.fc_img = nn.Sequential(
            nn.Linear(img_feat_dim, img_feat_dim),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(img_feat_dim, num_classes)
        )

        combined_dim = img_feat_dim + material_dim
        self.fc_comb = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(combined_dim, num_classes)
        )

    def forward(self, pixel_values, material=None):
        outputs = self.cnn(pixel_values=pixel_values)
        img_feats = outputs.pooler_output.view(outputs.pooler_output.size(0), -1)

        if material is None:
            return self.fc_img(img_feats)

        mat_emb = self.material_embedding(material)
        combined = torch.cat([img_feats, mat_emb], dim=1)
        return self.fc_comb(combined)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLO detector
yolo = YOLO("/home/sankalp/flake_classification/models/best.pt")
yolo.conf = 0.25

# Load classifier weights
ckpt = torch.load(
    "/home/sankalp/flake_classification/models/flake_classifier.pth",
    map_location=device
)
num_classes   = len(ckpt["class_to_idx"])
classifier    = FlakeLayerClassifier(
    num_materials=num_classes,
    material_dim=64,
    num_classes=num_classes,
    dropout_prob=0.1,
    freeze_cnn=False
).to(device)
classifier.load_state_dict(ckpt["model_state_dict"])
classifier.eval()

# Image processing transforms
clf_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

try:
    FONT = ImageFont.truetype("arial.ttf", 20)
except IOError:
    FONT = ImageFont.load_default()

# Inference + drawing
def detect_and_classify(image: Image.Image):
    img_rgb = np.array(image.convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1]
    results = yolo(img_bgr, device=str(device))
    boxes  = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    draw = ImageDraw.Draw(image)
    for (x1, y1, x2, y2), conf in zip(boxes, scores):
        crop = image.crop((x1, y1, x2, y2))
        inp  = clf_tf(crop).unsqueeze(0).to(device)  # (1,C,H,W)

        with torch.no_grad():
            logits = classifier(pixel_values=inp)
            pred   = logits.argmax(1).item()
            prob   = F.softmax(logits, dim=1)[0, pred].item()

        label = f"Layer {pred+1} ({prob:.2f})"
        # draw
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1-18)), label, fill="red", font=FONT)

    return image

# Gradio UI
demo = gr.Interface(
    fn=detect_and_classify,
    inputs=gr.Image(type="pil", label="Upload Flake Image"),
    outputs=gr.Image(type="pil", label="Annotated Output"),
    title="Flake Detection + Layer Classification",
    description="Upload an image → YOLO finds flakes → ResNet-18 head classifies their layer.",
)

if __name__ == "__main__":
    demo.launch(share=True)
