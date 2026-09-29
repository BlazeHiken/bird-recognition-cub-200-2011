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
    elif name == "clip_vit_b16":
        import open_clip
        clip_model = open_clip.create_model('ViT-B-16', pretrained='openai')
        
        class CLIPWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model.encode_image(x)
                
        model = CLIPWrapper(clip_model)
        feat_dim = 512
        norm = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        preprocess = transforms.Compose(base_transform + [norm])
    else:
        raise ValueError(f"Unknown model name: {name}")
        
    model.eval()
    return model, preprocess, feat_dim
