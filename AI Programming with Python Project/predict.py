import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json

# Define a function for parsing command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Predict the class of a flower.")
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the saved checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, help='Path to category-to-name mapping')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()

# Define a function to load the checkpoint
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if checkpoint['arch']== 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# Define a function to process the image
def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return preprocess(image).unsqueeze(0)

# Define a function to predict the class of an image
def predict(model, image_path, top_k, gpu, category_names=None):
    model.eval()
    image = process_image(image_path)
    
    if gpu and torch.cuda.is_available():
        model = model.cuda()
        image = image.cuda()
    else:
        model = model.cpu()
        image = image.cpu()
    
    with torch.no_grad():
        output = model(image)
    
    probabilities = torch.exp(output)
    top_p, top_class = probabilities.topk(top_k, dim=1)
    
    top_p = top_p.cpu().numpy().flatten()
    top_class = top_class.cpu().numpy().flatten()
    
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_class = [cat_to_name[str(c)] for c in top_class]
    
    return top_class, top_p

# Main function
def main():
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    
    top_class, top_p = predict(model, args.image_path, args.top_k, args.gpu, args.category_names)
    
    print("Top classes: ", top_class)
    print("Probabilities: ", top_p)

if __name__ == '__main__':
    main()
