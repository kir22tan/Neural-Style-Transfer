import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    return torch.mm(features, features.t()) / (c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target, weight=1.0):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()
        self.weight = weight

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = self.weight * nn.functional.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

content_layers = ["conv_4"]
style_layers = {
    "conv_1": 1.0,
    "conv_2": 0.75,
    "conv_3": 0.2,
    "conv_4": 0.2,
    "conv_5": 0.2,
}

def get_model_and_losses(cnn, style_img, content_img):
    normalization = Normalization([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])
    content_losses, style_losses = [], []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            weight = style_layers[name]
            style_loss = StyleLoss(target, weight)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[: i + 1]

    return model, content_losses, style_losses

def run_style_transfer(cnn, content_img, style_img,
                       num_steps=2000,
                       style_weight=1e9,
                       content_weight=1):
    model, content_losses, style_losses = get_model_and_losses(cnn, style_img, content_img)
    input_img = content_img.clone()
    optimizer = optim.Adam([input_img.requires_grad_()], lr=0.003)

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()
            run[0] += 1
            if run[0] % 400 == 0:
                print(f"Iteration {run[0]}: Style Loss {style_score.item():.4f} Content Loss {content_score.item():.4f}")
            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img
