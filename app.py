#!/usr/bin/env python
# coding: utf-8

# In[26]:


# Step 1: Installing Required Libraries
get_ipython().system('pip install transformers ftfy --quiet')
get_ipython().system('git clone https://github.com/stevejpapad/relevant-evidence-detection.git')
get_ipython().run_line_magic('cd', 'relevant-evidence-detection/src')


# In[27]:


# Step 2: Mounting Google Drive and Settting Paths
from google.colab import drive
drive.mount('/content/drive')

# Setting path for the all samples
base_path = "/content/drive/MyDrive/sample data"

# Setting path for RED-DOT Model Checkpoint
checkpoint_path = "/content/drive/MyDrive/checkpoints.pt"


# In[28]:


# Step 3: Import Required Libraries
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import math
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# In[29]:


# Step 4: Defining the Simple RED-DOT Classifier
class REDDOTClassifier(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


# In[32]:


# Step 5  Loading the model_state_dict from checkpoint
import torch

checkpoint_path = "/content/drive/MyDrive/checkpoints.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Checking that the expected structure exists
if "model_state_dict" not in checkpoint:
    raise KeyError("The checkpoint file does not contain a 'model_state_dict'. Please verify the file.")

# Loading the state_dict for transformer-based model
model_state = checkpoint["model_state_dict"]

# Printing sample keys
print("Sample keys from model_state_dict:")
for k in list(model_state.keys())[:10]:
    print(" •", k)


# In[33]:


# Step 6: Loading the CLIP ViT-L/14
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()


# In[34]:


# Step 7: Implementing the Helper - Entropy Calculation
def compute_entropy(prob):
    return -prob * math.log(prob + 1e-8) - (1 - prob) * math.log(1 - prob + 1e-8)


# In[35]:


# Step 8: Loop for All Samples and Run Inference
sample_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
results = []

for folder in sample_folders:
    folder_path = os.path.join(base_path, folder)

    # Loading image
    image = Image.open(os.path.join(folder_path, "Image.jpg")).convert("RGB")

    # Loading caption
    with open(os.path.join(folder_path, "caption.txt"), "r") as f:
        caption = f.read().strip()

    # Loading ground truth
    with open(os.path.join(folder_path, "GT.txt"), "r") as f:
        gt = f.read().strip()

    # Processing with CLIP
    inputs = clip_processor(text=[caption], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_embeds = clip_model.get_image_features(inputs["pixel_values"])
        text_embeds = clip_model.get_text_features(inputs["input_ids"])

    # Modality fusion
    combined_features = image_embeds + text_embeds

    # RED-DOT forward
    with torch.no_grad():
        logits = model(combined_features)
        prob = torch.sigmoid(logits).item()
        entropy = compute_entropy(prob)
        prediction = "Fake" if prob > 0.5 else "True"

    results.append({
        "Sample": folder,
        "Caption": caption,
        "Prediction": prediction,
        "Confidence": round(prob, 4),
        "Entropy": round(entropy, 4),
        "Ground Truth": gt
    })



# In[36]:


# Step 9: Show Final Results
df = pd.DataFrame(results)
from IPython.display import display
display(df)

