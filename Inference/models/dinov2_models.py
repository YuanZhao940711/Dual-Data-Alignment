import torch
import torch.nn as nn
import torch.hub

CHANNELS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

class DINOv2Model(nn.Module):
    def __init__(self, name, num_classes=1):
        super(DINOv2Model, self).__init__()
        print(f"Loading DINOv2 from hub: {name}")
        # self.model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)

        repo_path = "/root/autodl-tmp/AIGC_Detection/model_cache/dinov2_vitl14/facebookresearch_dinov2_main"

        self.model = torch.hub.load(
            repo_path,
            name,
            source="local",
            pretrained=False
        )

        ckpt_path = "/root/autodl-tmp/AIGC Detection/model_cache/dinov2_vitl14/checkpoints/dinov2_vitl14_pretrain.pth"

        state_dict = torch.load(ckpt_path,map_locatio="cpu")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]

        self.model.load_state_dict(state_dict)

        self.fc = nn.Linear(CHANNELS[name], num_classes)
    
    def forward(self, x, return_feature=False, return_tokens=False):
        if hasattr(self.model, 'forward_features'):
            features_dict = self.model.forward_features(x)
            features = features_dict['x_norm_clstoken']
        else:
            features = self.model(x)
            if isinstance(features, dict):
                features = features.get('x_norm_clstoken', features.get('last_hidden_state', None)[:, 0])
            
        if return_feature:
            return features, self.fc(features)
        
        return self.fc(features)
    
