
import streamlit as st
import subprocess
subprocess.call(["pip", "install", "-r", "./requirements.txt"])
import torch
from skimage.io import imread as imread
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

with open('./data/allergens.txt', 'r') as file:
    allergens = [line.strip() for line in file]

class FineTunedResNet(nn.Module):
    def __init__(self, num_classes=len(allergens)):
        super(FineTunedResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = FineTunedResNet()
model.load_state_dict(torch.load('./ResNet50_allergens_model_1e-3.pth', map_location='cpu'))
model.cpu()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("lxhyylian-Food Allergens Recognition")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred_title = ', '.join(['{} ({:2.1f}%)'.format(allergens[j], 100 * torch.sigmoid(output[0, j]).item())
                            for j, v in enumerate(output.squeeze())
                            if torch.sigmoid(v) > 0.5])

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Predicted allergens in food: ", pred_title)
