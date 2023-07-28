from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier

class ConvNetClassifier(ImageClassifier):

    image_processing = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1302,), (0.3069,))
    ])

    def postprocess(self, output):
        return output.argmax(1).tolist()