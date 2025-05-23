![3.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/oN-EYfwR1vvdgHkVKAOKP.png)

# open-age-detection

> `open-age-detection` is a vision-language encoder model fine-tuned from `google/siglip2-base-patch16-512` for **multi-class image classification**. It is trained to classify the estimated age group of a person from an image. The model uses the `SiglipForImageClassification` architecture.

> \[!note]
> *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*
> [https://arxiv.org/pdf/2502.14786](https://arxiv.org/pdf/2502.14786)

```py
Classification Report:
                  precision    recall  f1-score   support

      Child 0-12     0.9827    0.9859    0.9843      2193
  Teenager 13-20     0.9663    0.8713    0.9163      1779
     Adult 21-44     0.9669    0.9884    0.9775      9999
Middle Age 45-64     0.9665    0.9538    0.9601      3785
        Aged 65+     0.9737    0.9706    0.9722      1260

        accuracy                         0.9691     19016
       macro avg     0.9713    0.9540    0.9621     19016
    weighted avg     0.9691    0.9691    0.9688     19016
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/ngGPmqGLlqQnfwRhy8phZ.png)

---

## Label Space: 5 Age Groups

```
Class 0: Child 0–12  
Class 1: Teenager 13–20  
Class 2: Adult 21–44  
Class 3: Middle Age 45–64  
Class 4: Aged 65+
```

---

## Install Dependencies

```bash
pip install -q transformers torch pillow gradio hf_xet
```

---

## Inference Code

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/open-age-detection"  # Updated model name
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Updated label mapping
id2label = {
    "0": "Child 0-12",
    "1": "Teenager 13-20",
    "2": "Adult 21-44",
    "3": "Middle Age 45-64",
    "4": "Aged 65+"
}

def classify_image(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=5, label="Age Group Detection"),
    title="open-age-detection",
    description="Upload a facial image to estimate the age group: Child, Teenager, Adult, Middle Age, or Aged."
)

if __name__ == "__main__":
    iface.launch()
```

---

## Demo Inference

![Screenshot 2025-05-20 at 21-04-41 open-age-detection.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/V-kG5yBm3F1uixB501XML.png)
![Screenshot 2025-05-20 at 21-49-28 open-age-detection.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/n6HyF0E_hvvV1QjMRCxd3.png)
![Screenshot 2025-05-20 at 21-50-03 open-age-detection.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/8c5n2UcC7ICmfui6q5k6F.png)
![Screenshot 2025-05-20 at 21-56-22 open-age-detection.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/I1l7x55WvlibUKmKQNvGn.png)
![Screenshot 2025-05-20 at 21-58-09 open-age-detection.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/5KHykSEnBtJy1cZePfveO.png)

---

## Intended Use

`open-age-detection` is designed for:

* **Demographic Analysis** – Estimate age groups for statistical or analytical applications.
* **Smart Personalization** – Age-based content or product recommendation.
* **Access Control** – Assist systems requiring age verification.
* **Social Research** – Study age-related trends in image datasets.
* **Surveillance and Security** – Profile age ranges in monitored environments.
