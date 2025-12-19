import torch
import torch.nn as nn
import torchvision.models.video as models

class TwoStreamNetwork(nn.Module):
    def __init__(self):
        super(TwoStreamNetwork, self).__init__()
        
        # Stream 1: RGB
        self.rgb_backbone = models.r3d_18(weights=None)
        self.rgb_backbone.fc = nn.Identity() # Remove classification head
        
        # Stream 2: Optical Flow
        self.flow_backbone = models.r3d_18(weights=None)
        self.flow_backbone.fc = nn.Identity()
        
        # Fusion
        # R3D_18 output dim is 512
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
    def forward(self, rgb, flow):
        idx_rgb = self.rgb_backbone(rgb)
        idx_flow = self.flow_backbone(flow)
        
        combined = torch.cat((idx_rgb, idx_flow), dim=1)
        out = self.fusion_fc(combined)
        return out
