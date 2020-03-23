import argparse
import numpy as np
import pylab as plt

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from kernel_conv.utils_kernels import extract_tensor_patches
from kernel_conv.conv import KernelConvUnfold2d
from kernel_conv.conv import KernelConvConv2d

# from kernel_conv.conv import kernel_wrapper
from kernel_conv.kernels import MaskedDCTKernel
# from tqdm import tqdm

from kernel_conv.my_dct_pytorch import compute_dct_pytorch

torch.autograd.set_detect_anomaly(True)

class SharpnessModelConv(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.kernel_size = 50
        self.kernel_fn = MaskedDCTKernel(in_features=self.kernel_size, cuda=False)

        # self.pad_layer = torch.nn.modules.ReflectionPad2d([0, self.kernel_size // 2, 0, self.kernel_size // 2])
        self.dct_conv = nn.Sequential(
            KernelConvConv2d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, kernel_fn=self.kernel_fn,
                             stride=4, padding=25, dilation=1, groups=1, bias=False, padding_mode='reflect')
        )

        self.fc = nn.Sequential(
        nn.Linear(1, num_classes)
        )

    def forward(self, x):
        # x = self.pad_layer(x)
        x = self.dct_conv(x).abs()
        # print(x)
        # x = torch.mean(x, dim=3)
        # x, _ = torch.max(x, dim=2)
        # x = self.fc(x)
        return x #[self.kernel_size // 2:, self.kernel_size // 2:]


class SharpnessModelUnfold(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.kernel_size = 50
        self.kernel_fn = MaskedDCTKernel(in_features=self.kernel_size, cuda=False)
        self.dct_conv = nn.Sequential(
            KernelConvUnfold2d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, kernel_fn=self.kernel_fn,
                         stride=4, padding=25, dilation=1, groups=1, bias=False, padding_mode='reflect')
        )

        self.fc = nn.Sequential(
        nn.Linear(1, num_classes)
        )

    def forward(self, x):
        # x = self.pad_layer(x)
        x = self.dct_conv(x).abs()
        # print(x)
        # x = torch.mean(x, dim=3)
        # x, _ = torch.max(x, dim=2)
        # x = self.fc(x)
        return x #[self.kernel_size // 2:, self.kernel_size // 2:]


def main():
    # # Parsing command line args
    # parser = argparse.ArgumentParser(description='CIFAR10 example')
    # parser.add_argument('--kernel', type=str, default=None,
    #                     help='Kernel type to use: [gaussian, polynomial, sigmoid, DCT] (default: None)')
    #
    # parser.add_argument('--epoch', type=int, default=2, help='Number of epochs (default: 2)')
    # parser.add_argument('--batch_size', type=int, default=4, help='Batch suze (default: 4)')
    # parser.add_argument('--gpu', type=bool, default=True, help='Use GPU? (default: True)')
    #
    # args = parser.parse_args()
    #
    # device = 'cpu'
    # if args.gpu:
    #     device = 'cuda'
    #
    # # Initiating network
    # resnet50 = torchvision.models.resnet50()
    # resnet50._modules['fc'] = torch.nn.Linear(2048, 10, True)
    # net = resnet50
    #
    # if args.kernel == 'gaussian':
    #     kernel_wrapper(net, GaussianKernel())
    # elif args.kernel == 'polynomial':
    #     kernel_wrapper(net, PolynomialKernel())
    # elif args.kernel == 'sigmoid':
    #     kernel_wrapper(net, SigmoidKernel())
    # elif args.kernel == "DCT":
    #     kernel_wrapper(net, SigmoidKernel())
    # elif args.kernel is not None:
    #     raise Exception('Invalid kernel')

    net = SharpnessModelUnfold(num_classes=3)
    net2 = SharpnessModelConv(num_classes=3)

    # net.to(device)

    # Loading datasets
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
        # transforms.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26))
    ])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    testset = torchvision.datasets.FakeData(transform=transform)
    # testset = torchvision.datasets.(root='./data', train=False, download=True, transform=transform)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    img = plt.imread("./resourses/image_blur3.jpg")[:200, :200, 0:1]
    torch_images = torch.Tensor(img).transpose(0, 2).transpose(1, 2).unsqueeze(0)
    with torch.no_grad():
        net.eval()
        # print(net)
        # for (images, labels) in testloader:
            # images = images.to(device)
            # labels = labels.to(device)
        outputs = net(torch_images)
        outputs2 = net2(torch_images)

    # dct_map_2 = compute_dct_pytorch(img[:, :, 0].astype(np.float))
    # dct_map_2[0][0, 0] = dct_map[0].max()
    # print(np.abs(dct_map).mean())
    # print(np.abs(dct_map_2[0]).mean())
    # print(np.abs(dct_map_conv).mean())

    dct_map = outputs.data.numpy()[0, 0]
    dct_map_conv = outputs2.data.numpy()[0, 0]

    assert outputs.shape == outputs2.shape
    assert np.abs(dct_map_conv).mean() == np.abs(dct_map).mean()
    assert np.abs(dct_map_conv).max() == np.abs(dct_map).max()

    # print(np.abs(dct_map).max())
    # print(np.abs(dct_map_2[0]).max())
    # print(np.abs(dct_map_conv).max())


    # print(np.abs(dct_map).min())
    # print(np.abs(dct_map_2[0]).min())
    # print(np.abs(dct_map_conv).min())


    # image = torch_images.data.numpy()[0, 0]

    # print(image.shape)
    # print(dct_map.shape)
    # print(dct_map_2[0].shape)
    # print(dct_map_conv.shape)

    # plt.show()
    # print(torch_images.shape)
    # patches = extract_tensor_patches(torch_images, window_size=50, stride=4)
    # patches_unfold = F.unfold(torch_images, kernel_size=50, stride=4)
    # print(patches.shape)
    # print(patches_unfold.shape)

    # print(patches[0])
    # print(patches_unfold[0])

    # kconv = KernelConv2d(in_channels=1, out_channels=1, kernel_size=50, kernel_fn=self.kernel_fn,
    #                      stride=4, padding=0, dilation=1, groups=1, bias=False, padding_mode='reflection')
    # patches_old = kconv(torch_images, window_size=50, stride=4)
    # plt.imsave("dct_map_2.png", dct_map_2[0])
    # plt.imsave("dct_map.png", dct_map)
    # plt.imsave("dct_map_conv.png", dct_map_conv)
    #
    # plt.imsave("image.png", image)


if __name__ == '__main__':
    main()
