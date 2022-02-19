import torch.nn as nn

MODEL_FILE = 'vae.pt'


def conv_sizes(N, H, W, kernel=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
    hout, wout = H, W
    sizes = [(hout, wout)]

    for _ in range(1, N + 1):
        hout = (hout + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1
        wout = (wout + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1
        sizes.append((int(hout), int(wout)))
    return sizes


def deconv_sizes(N, H, W, kernel=(3, 3), stride=(1, 1), padding=(0, 0), output_padding=(0, 0), dilation=(1, 1)):
    hout, wout = H, W
    sizes = [(hout, wout)]

    for _ in range(1, N + 1):
        hout = (hout - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel[0] - 1) + output_padding[0] + 1
        wout = (wout - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel[1] - 1) + output_padding[1] + 1
        sizes.append((hout, wout))
    return sizes


class PrintShape(nn.Module):
    def forward(self, input):
        # print(f'layer shape={input.shape}')
        return input


class PrintExample(nn.Module):
    def forward(self, input):
        batch_i, channel_i = 0, 0
        # print('Example=', input[batch_i][channel_i])
        return input


class Flatten(nn.Module):
    def forward(self, input):
        batch_size = input.size(0)
        return input.view(batch_size, -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1, 1, 1)


class Review(nn.Module):
    def __init__(self, channel, sizes):
        super().__init__()
        self.channel, self.sizes = channel, sizes

    def forward(self, input):
        batch_size = input.size(0)
        return input.view(batch_size, self.channel, self.sizes[0], self.sizes[1])

def video_gen(meta):
    "meta is list of images"