import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
from torchvision.models.feature_extraction import create_feature_extractor
 
def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2))/(c*h*w)
 
class VGGLoss(nn.Module):
    def _init_(self) -> None:
        super()._init_()
        layer_ids = [22, 32, 42]
        self.weights = [5, 15, 2]
        m = vgg16_bn(pretrained=True).features.eval()
        return_nodes = {f'{x}': f'feat{i}' for i, x in enumerate(layer_ids)}
        self.vgg_fx = create_feature_extractor(m, return_nodes=return_nodes)
        self.vgg_fx.requires_grad_(False)
        self.l1_loss = nn.L1Loss()
 
    def forward(self, x, y):
        x_vgg = self.vgg_fx(x)
        with torch.inference_mode():
            y_vgg = self.vgg_fx(y)
        loss = self.l1_loss(x, y)
        for i, k in enumerate(x_vgg.keys()):
            loss += self.weights[i] * self.l1_loss(x_vgg[k], y_vgg[k].detach_())       # feature loss
            loss += self.weights[i]**2 * 5e3 * self.l1_loss(gram_matrix(x_vgg[k]), gram_matrix(y_vgg[k]))  # style loss
        return loss

       
