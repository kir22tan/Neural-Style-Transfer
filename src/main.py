from utils import image_loader, imshow
from model import run_style_transfer
import torchvision.models as models
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content_path = "path_to_content.jpg"
style_path = "path_to_style.jpg"

content_img = image_loader(content_path)
style_img = image_loader(style_path, (content_img.shape[-1], content_img.shape[-2]))

assert content_img.size() == style_img.size(), "Style and content images must be the same size!"

cnn = models.vgg19(pretrained=True).features.to(device).eval()

output = run_style_transfer(cnn, content_img, style_img)
imshow(output, title="Output Image")
