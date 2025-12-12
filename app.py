import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
from matplotlib import cm

# ---------------------------------------
# 1. BASIC CONFIG
# ---------------------------------------
st.set_page_config(page_title="Caltech256 Grad-CAM Demo", layout="centered")
st.title("🔍 Caltech-256 Grad-CAM (ResNet-18)")
st.write("Upload an image from the 10 selected Caltech-256 classes to see prediction + Grad-CAM.")

device = torch.device("cpu")

# class names from your notebook (first 10 classes)
CLASS_NAMES = [
    "001.ak47",
    "002.american-flag",
    "003.backpack",
    "004.baseball-bat",
    "005.baseball-glove",
    "006.basketball-hoop",
    "007.bat",
    "008.bathtub",
    "009.bear",
    "010.beer-mug",
]

# same normalization you used in the notebook
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def denormalize(img_tensor):
    """
    img_tensor: 3 x H x W (PyTorch tensor, normalized)
    returns: H x W x 3 (numpy image in [0,1])
    """
    img = img_tensor.detach().cpu().numpy()
    img = img * imagenet_std + imagenet_mean
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))
    return img


# ---------------------------------------
# 2. LOAD MODEL (NO DOWNLOAD, JUST YOUR WEIGHTS)
# ---------------------------------------
@st.cache_resource
def load_model_and_gradcam():
    # build same architecture as training, but avoid downloading weights
    model = models.resnet18(weights=None)
    num_classes = len(CLASS_NAMES)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    state_dict = torch.load("resnet18_caltech256.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Grad-CAM hooks on last conv block
    target_layer = model.layer4[-1].conv2

    activations = {"value": None}
    gradients = {"value": None}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    def gradcam_fn(input_tensor, class_idx=None):
        """
        input_tensor: 1 x 3 x H x W
        returns: heatmap (H x W, numpy in [0,1]), predicted class idx
        """
        activations["value"] = None
        gradients["value"] = None

        output = model(input_tensor)  # 1 x num_classes

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        model.zero_grad()
        score.backward(retain_graph=True)

        # [1, C, H', W']
        grads = gradients["value"]
        acts = activations["value"]

        alpha = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (alpha * acts).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        cam = torch.relu(cam)

        # upsample to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()  # H x W
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, class_idx, output.detach()

    return model, gradcam_fn


model, gradcam = load_model_and_gradcam()

# ---------------------------------------
# 3. UI: FILE UPLOAD
# ---------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an image (jpg / png)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(pil_img, use_column_width=True)

    # preprocess
    img_tensor = preprocess(pil_img)
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # Grad-CAM + prediction
    heatmap, pred_idx, logits = gradcam(input_tensor)
    probs = torch.softmax(logits, dim=1)[0]
    pred_prob = probs[pred_idx].item()

    # build overlay using denormalized model input
    orig_img = denormalize(img_tensor)          # 224 x 224 x 3 in [0,1]
    heatmap_rgb = cm.jet(heatmap)[..., :3]      # apply colormap
    overlay = 0.6 * orig_img + 0.4 * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)

    # top-3 predictions
    topk_prob, topk_idx = torch.topk(probs, k=3)
    st.subheader("Predictions")
    for i in range(3):
        cls_name = CLASS_NAMES[topk_idx[i].item()]
        st.write(f"**{i+1}. {cls_name}** — {topk_prob[i].item():.3f}")

    st.subheader("Grad-CAM Overlay")
    st.image(
        overlay,
        caption=f"Predicted: {CLASS_NAMES[pred_idx]} (prob={pred_prob:.3f})",
        use_column_width=True,
    )
else:
    st.info("👆 Upload an image to see the prediction and Grad-CAM heatmap.")
