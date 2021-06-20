import torch

from utils.util import opt_get

levelToName = {
    0: 'fea_up8',
    1: 'fea_up4',
    2: 'fea_up2',
    3: 'fea_up1',
    4: 'fea_up0'
}


def get_named_level_conditionals(opt, gt, rrdbResults):
    levelConditionalOpt = opt_get(opt, ['network_G', 'flow', 'levelConditional'])

    if levelConditionalOpt is None:
        name_level_conditionals = rrdbResults

    elif levelConditionalOpt['type'] == "rgb":
        l4_image = torch.nn.functional.interpolate(gt, scale_factor=1 / 2 ** 4, mode='bilinear')
        l4_conds = [
            torch.nn.functional.interpolate(l4_image, scale_factor=2 ** 4, mode='bilinear'),
            torch.nn.functional.interpolate(l4_image, scale_factor=2 ** 3, mode='bilinear'),
            torch.nn.functional.interpolate(l4_image, scale_factor=2 ** 2, mode='bilinear'),
            torch.nn.functional.interpolate(l4_image, scale_factor=2 ** 1, mode='bilinear'),
            l4_image
        ]

        level_conditionals = [
            None,
            torch.cat([rrdbResults[levelToName[1]], l4_conds[1]], dim=1),
            torch.cat([rrdbResults[levelToName[2]], l4_conds[2]], dim=1),
            torch.cat([rrdbResults[levelToName[3]], l4_conds[3]], dim=1),
            torch.cat([rrdbResults[levelToName[4]], l4_conds[4]], dim=1),
        ]

        name_level_conditionals = {}
        for idx, level_conditional in enumerate(level_conditionals):
            name_level_conditionals[levelToName[idx]] = level_conditionals[idx]
    else:
        raise RuntimeError("Not found")

    return name_level_conditionals
