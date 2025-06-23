import gradio as gr
import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo_flake.pt')
classifier_model = torch.load('resnet18_flake_classifier.pth', map_location=torch.device('cpu'))
classifier_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = ['1-layer', '2-layer', '3-layer', '4-layer']

# Inference pipeline
def analyze_image(image):
    image_np = np.array(image)
    results = yolo_model(image_np)

    annotated_image = results.render()[0]
    flake_outputs = []

    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        flake_crop = image_np[y1:y2, x1:x2]
        flake_pil = Image.fromarray(flake_crop)
        input_tensor = transform(flake_pil).unsqueeze(0)

        with torch.no_grad():
            pred = classifier_model(input_tensor)
            pred_label = class_names[pred.argmax(dim=1).item()]
        
        flake_outputs.append({
            "coords": (x1, y1, x2, y2),
            "label": pred_label,
            "confidence": round(conf.item(), 2)
        })

        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_image, pred_label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return Image.fromarray(annotated_image), flake_outputs

iface = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(label="Detection & Classification"),
        gr.JSON(label="Details")
    ],
    title="Flake Detection and Layer Classification",
    description="Upload an image of a flake. The system will detect flakes and classify their layer type (e.g., mono, bi, multi-layer)."
)

if __name__ == "__main__":
    iface.launch()
