import torch
import random
from torch import nn as nn

from models.modules.flow import squeeze2d, unsqueeze2d


def doShiftW(channel, shift_w):
    shift_abs = abs(shift_w)
    if shift_w == 0:
        return channel
    if shift_w < 0:
        return torch.cat([channel[:, :, shift_abs:], channel[:, :, 0:shift_abs]], dim=2)
    if shift_w > 0:
        return torch.cat([channel[:, :, -shift_abs:], channel[:, :, :-shift_abs]], dim=2)


def doShiftH(channel, shift_h):
    shift_abs = abs(shift_h)
    if shift_h == 0:
        return channel
    if shift_h < 0:
        return torch.cat([channel[:, shift_abs:], channel[:, 0:shift_abs]], dim=1)
    if shift_h > 0:
        return torch.cat([channel[:, -shift_abs:], channel[:, :-shift_abs]], dim=1)


def doShift(channel, shift_h, shift_w):
    return doShiftW(doShiftH(channel, shift_h), shift_w)


class Shift(nn.Module):
    def __init__(self, shifts):
        super().__init__()
        self.shifts = shifts

    def forward(self, input: torch.Tensor, logdet=None, reverse=False):
        if not reverse:
            channels = []
            for channel_idx, (shift_h, shift_w) in enumerate(self.shifts):
                channels.append(doShift(input[:, channel_idx], shift_h, shift_w))
            output = torch.stack(channels, dim=1)
        else:
            channels = []
            for channel_idx, (shift_h, shift_w) in enumerate(self.shifts):
                channels.append(doShift(input[:, channel_idx], -shift_h, -shift_w))
            output = torch.stack(channels, dim=1)
        return output, logdet


class SqueezeShift(nn.Module):
    def __init__(self, shifts):
        super().__init__()
        self.shifter = Shift(shifts)

    def forward(self, z: torch.Tensor, logdet=None, reverse=False):
        if not reverse:
            z = unsqueeze2d(z)
            z, _ = self.shifter(z, reverse=False)
            z = squeeze2d(z)
        else:
            z = unsqueeze2d(z)
            z, _ = self.shifter(z, reverse=True)
            z = squeeze2d(z)
        return z, logdet


def random_shift():
    return (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))


def get_random_shifts(K, channels):
    shifts_list = []
    for _ in range(K):
        shifts = []
        for _ in range(channels):
            shifts.append(random_shift())
        shifts_list.append(shifts)

    shifts = []
    for _ in range(channels):
        shifts.append((0, 0))
    shifts_list.append(shifts)

    return shifts_list


def shift_list_to_sequence(shifts_list):
    shifts_sequence = []
    shifts_sequence.append(shifts_list[0])
    for shift_idx in range(1, len(shifts_list)):
        next_shift = [(new_s[0] - previouse_s[0], new_s[1] - previouse_s[1])
                      for new_s, previouse_s in zip(shifts_list[shift_idx], shifts_list[shift_idx - 1])]
        shifts_sequence.append(next_shift)
    return shifts_sequence
