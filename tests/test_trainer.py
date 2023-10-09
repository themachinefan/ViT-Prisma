import unittest
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss
from vit_planetarium.models.base_vit import BaseViT  # Assuming you have a ViT model defined somewhere

from vit_planetarium.configs.MNISTConfig import Config
from vit_planetarium.training.trainer import train

class TestTrainingFunction(unittest.TestCase):

        # Load MNIST dataset
        def setUp(self):
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self.train_dataset = datasets.MNIST(root='../../data', train=True, transform=transform, download=True)
            self.val_dataset = datasets.MNIST(root='../../data', train=False, transform=transform, download=True)

            self.config = Config()

            # Define a simple model
            self.model = BaseViT(self.config)

        def test_train_function(self):
            trained_model = train(self.model, self.config, self.train_dataset, self.val_dataset)
            self.assertIsInstance(trained_model, type(self.model))

if __name__ == '__main__':
    unittest.main()
