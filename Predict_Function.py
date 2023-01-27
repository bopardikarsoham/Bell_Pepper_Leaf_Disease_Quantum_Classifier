from torchvision import models, transforms
import torch
from PIL import Image
from Quantum_Model_Function import Quantumnet


def predict(image_path):
    device = torch.device("cpu")
    model_hybrid = models.mobilenet_v2(pretrained=True)
    model_hybrid.classifier[1] = Quantumnet()

    # load model
    PATH = "bell_pepper_quantum.pt"
    model_hybrid.load_state_dict(torch.load(PATH))


    #https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    model_hybrid.eval()
    out = model_hybrid(batch_t)

    with open('classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
