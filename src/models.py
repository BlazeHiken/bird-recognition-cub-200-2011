import torch
import timm
from torchvision import transforms

def load_backbone(name="resnet50"):
    # Shared resize policy
    base_transform = [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    
    if name == "resnet50":
        model = timm.create_model("resnet50", pretrained=True, num_classes=0)
        feat_dim = model.num_features
        # Native normalization
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose(base_transform + [norm])
    elif name == "dinov2_vits14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        feat_dim = 384
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose(base_transform + [norm])
    else:
        raise ValueError(f"Unknown model name: {name}")
        
    model.eval()
    return model, preprocess, feat_dim
