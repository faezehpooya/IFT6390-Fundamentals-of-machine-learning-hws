import random
import os
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, normalize, affine
from PIL import Image
from typing import Tuple, List, NamedTuple
from tqdm import tqdm
import zipfile
from urllib import request


# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)


class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    dense_hiddens: Tuple[int, ...] = (256, 256)


class Trainer:
    def __init__(self,
                 network_type: str = "mlp",
                 net_config: NetworkConfiguration = NetworkConfiguration(),
                 lr: float = 0.001,
                 batch_size: int = 128,
                 activation_name: str = "relu"):
        self.train, self.test = self.load_dataset()
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        input_dim = self.train[0].shape[1:]
        if network_type == "mlp":
            self.network = self.create_mlp(input_dim[0]*input_dim[1]*input_dim[2], 
                                           net_config,
                                           activation_function)
        elif network_type == "cnn":
            self.network = self.create_cnn(input_dim[0], 
                                           net_config, 
                                           activation_function)
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.lr = lr
        self.batch_size = batch_size

        self.train_logs = {'train_loss': [], 'test_loss': [],
                           'train_mae': [], 'test_mae': []}

    @staticmethod
    def load_dataset() -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if not os.path.exists('./rotated_fashion_mnist'):
            url = 'https://drive.google.com/u/0/uc?id=1NQPmr01eIafQKeH9C9HR0lGuB5z6mhGb&export=download&confirm=t&uuid=645ff20a-d47b-49f0-ac8b-4a7347529c8e&at=AHV7M3d_Da0D7wowJlTzzZxDky5c:1669325231545'
            with request.urlopen(url) as f:
                with open('./rotated_fashion_mnist.zip', 'wb') as out:
                    out.write(f.read())
            with zipfile.ZipFile('./rotated_fashion_mnist.zip', 'r') as zip_ref:
                zip_ref.extractall()
            os.remove('./rotated_fashion_mnist.zip')
            
        datapath = './rotated_fashion_mnist'

        def get_paths_and_rots(split: str) -> List[Tuple[str, float]]:
            image_paths, rots = [], []
            files = os.listdir(os.path.join(datapath, split))
            for file in files:
                image_paths.append(os.path.join(datapath, split, file))
                rots.append(float(file.split('_')[1].split('.')[0]))
            return image_paths, rots
        
        def to_tensors(image_paths: List[str], rots: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
            images = [normalize(to_tensor(Image.open(path)), (0.5,), (0.5,)) 
                      for path in image_paths]
            images = torch.stack(images)
            labels = torch.tensor(rots).view(-1, 1)
            return images, labels

        X_train, y_train = to_tensors(*get_paths_and_rots('train'))
        X_test, y_test = to_tensors(*get_paths_and_rots('test'))
        
        # Normalize y for easier training
        mean, std = y_train.mean(), y_train.std()
        y_train = (y_train - mean) / std
        y_test = (y_test - mean) / std
        
        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def create_mlp(input_dim: int, net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a multi-layer perceptron (MLP) network.

        :param net_config: a NetworkConfiguration named tuple. Only the field 'dense_hiddens' will be used.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the MLP.
        """
        # TODO write code here
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=input_dim, out_features=net_config.dense_hiddens[0]),
            activation,
        )
        for i in range(len(net_config.dense_hiddens)):
            if i != len(net_config.dense_hiddens) - 1:
                model.append(
                    torch.nn.Linear(in_features=net_config.dense_hiddens[i], out_features=net_config.dense_hiddens[i+1]),
                )
                model.append(activation)
            else:
                model.append(
                    torch.nn.Linear(in_features=net_config.dense_hiddens[i], out_features=1),
                )

        return model

    @staticmethod
    def create_cnn(in_channels: int, net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a convolutional network.

        :param in_channels: The number of channels in the input image.
        :param net_config: a NetworkConfiguration specifying the architecture of the CNN.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the CNN.
        """
        # TODO write code here
        #model = torch.nn.Sequential(
        #    #first layer
        #    torch.nn.Conv2d(
        #        in_channels,
        #        net_config.n_channels[0],
        #        kernel_size=net_config.kernel_sizes[0],
        #        stride=net_config.strides[0]
        #    ),
        #    activation,
        #    torch.nn.MaxPool2d(kernel_size=2),

            #second layer
        #    torch.nn.Conv2d(
        #        net_config.n_channels[0],
        #        net_config.n_channels[1],
        #        kernel_size=net_config.kernel_sizes[1],
        #        stride=net_config.strides[1]
        #    ),
        #    activation,
        #    torch.nn.MaxPool2d(kernel_size=2),

            #third layer
         #   torch.nn.Conv2d(
        #        net_config.n_channels[1],
        #        net_config.n_channels[2],
        #        kernel_size=net_config.kernel_sizes[2],
        #        stride=net_config.strides[2]
         #   ),
        #    torch.nn.AdaptiveMaxPool2d((4, 4)),
        #    torch.nn.Flatten(),
        #    torch.nn.Linear(in_features=net_config.n_channels[2]*4*4, out_features=net_config.dense_hiddens[0]),
        #    activation,
        #    torch.nn.Linear(in_features=net_config.dense_hiddens[0], out_features=net_config.dense_hiddens[1]),
        #    activation,
        #    torch.nn.Linear(in_features=net_config.dense_hiddens[1], out_features=1),
        #)


        model =torch.nn.Sequential(
            #first layer
            torch.nn.Conv2d(
                in_channels,
                net_config.n_channels[0],
                kernel_size=net_config.kernel_sizes[0],
                stride=net_config.strides[0]
            ),
            activation,
            torch.nn.MaxPool2d(kernel_size=2),
        )

        for i in range(len(net_config.n_channels)):
            if i != len(net_config.n_channels) - 2:
                model.append(
                    torch.nn.Conv2d(
                        net_config.n_channels[i],
                        net_config.n_channels[i+1],
                        kernel_size=net_config.kernel_sizes[i+1],
                        stride=net_config.strides[i+1]
                    )
                )
                model.append(activation)
                model.append(torch.nn.MaxPool2d(kernel_size=2))

            else:
                model.append(
                    torch.nn.Conv2d(
                        net_config.n_channels[i],
                        net_config.n_channels[i+1],
                        kernel_size=net_config.kernel_sizes[i+1],
                        stride=net_config.strides[i+1]
                    )
                )
                model.append(torch.nn.AdaptiveMaxPool2d((4, 4)))
                model.append(torch.nn.Flatten())
                model.append(
                    torch.nn.Linear(
                        in_features=net_config.n_channels[i+1]*4*4,
                        out_features=net_config.dense_hiddens[0])
                )
                break

        for i in range(len(net_config.dense_hiddens)):
            if i < len(net_config.dense_hiddens) - 1:
                model.append(
                    torch.nn.Linear(
                        in_features=net_config.dense_hiddens[i],
                        out_features=net_config.dense_hiddens[i + 1]
                    )
                )
                model.append(activation)
            else:
                print('hello form the last layers')
                model.append(
                    torch.nn.Linear(in_features=net_config.dense_hiddens[i], out_features=1),
                )

        return model

    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        # TODO WRITE CODE HERE

        if activation_str == 'relu':
            return torch.nn.ReLU()

        elif activation_str == 'sigmoid':
            return torch.nn.Sigmoid()

        else:
            return torch.nn.Tanh()

    def compute_loss_and_mae(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO WRITE CODE HERE
        mae = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()
        mse_output = mse(self.network(X), y)
        mae_output = mae(self.network(X), y)
        return mse_output, mae_output

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        # TODO WRITE CODE HERE
        self.optimizer.zero_grad()
        loss = self.compute_loss_and_mae(X_batch, y_batch)[0]
        loss.backward()
        self.optimizer.step()

    def log_metrics(self, X_train: torch.Tensor, y_train: torch.Tensor,
                    X_test: torch.Tensor, y_test: torch.Tensor) -> None:
        self.network.eval()
        with torch.inference_mode():
            train_loss, train_mae = self.compute_loss_and_mae(X_train, y_train)
            test_loss, test_mae = self.compute_loss_and_mae(X_test, y_test)
        self.train_logs['train_mae'].append(train_mae.item())
        self.train_logs['test_mae'].append(test_mae.item())
        self.train_logs['train_loss'].append(train_loss.item())
        self.train_logs['test_loss'].append(test_loss.item())

    def train_loop(self, n_epochs: int):
        # Prepare train and validation data
        X_train, y_train = self.train
        X_test, y_test = self.test

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        self.log_metrics(X_train[:2000], y_train[:2000], X_test, y_test)
        for epoch in tqdm(range(n_epochs)):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                self.training_step(minibatchX, minibatchY)
            self.log_metrics(X_train[:2000], y_train[:2000], X_test, y_test)
        return self.train_logs

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO WRITE CODE HERE

        mse_output, mae_output = self.compute_loss_and_mae(X, y)
        return torch.mean(mse_output), mae_output


    def test_equivariance(self):
        from functools import partial
        test_im = (self.train[0][0] + 1) / 2
        conv = torch.nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, stride=1, padding=0)
        fullconv_model = lambda x: torch.relu(conv((torch.relu(conv((x))))))
        model = fullconv_model

        shift_amount = 5
        shift = partial(affine, angle=0, translate=(shift_amount, shift_amount), scale=1, shear=0)
        rotation = partial(affine, angle=90, translate=(0, 0), scale=1, shear=0)

        # TODO CODE HERE
        pass
