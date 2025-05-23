import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image_name, target_size=None):
    image = Image.open(image_name).convert("RGB")
    if target_size:
        image = image.resize(target_size, Image.LANCZOS)
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
