import numpy as np
import torch
import torch.nn as nn


def parse_cfg(cfg_path):
    """Parse a darknet .cfg file into an ordered list of block dicts."""
    with open(cfg_path, "r") as f:
        lines = f.read().split("\n")
    lines = [x.strip() for x in lines if x.strip() and not x.strip().startswith("#")]

    blocks = []
    block = None
    for line in lines:
        if line.startswith("["):
            if block is not None:
                blocks.append(block)
            block = {"type": line[1:-1].strip()}
        else:
            key, _, value = line.partition("=")
            block[key.strip()] = value.strip()
    if block is not None:
        blocks.append(block)
    return blocks


class Upsample(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.stride, mode="nearest")


def create_modules(blocks):
    """Build the module list from cfg blocks. YOLO/shortcut/route carry config only."""
    module_list = nn.ModuleList()
    yolo_blocks = []
    output_filters = [int(blocks[0]["channels"])]

    for i, block in enumerate(blocks[1:], start=1):
        block_type = block["type"]
        filters = output_filters[-1]
        module = None

        if block_type == "convolutional":
            bn = int(block.get("batch_normalize", 0))
            out_filters = int(block["filters"])
            kernel = int(block["size"])
            stride = int(block["stride"])
            pad = int(block.get("pad", 0))
            padding = (kernel - 1) // 2 if pad else 0
            activation = block.get("activation", "leaky")

            layers = nn.Sequential()
            layers.add_module(
                "conv",
                nn.Conv2d(output_filters[-1], out_filters, kernel, stride, padding, bias=not bn),
            )
            if bn:
                layers.add_module("bn", nn.BatchNorm2d(out_filters))
            if activation == "leaky":
                layers.add_module("leaky", nn.LeakyReLU(0.1, inplace=True))
            module = layers
            filters = out_filters

        elif block_type == "maxpool":
            kernel = int(block["size"])
            stride = int(block["stride"])
            padding = (kernel - 1) // 2 if int(block.get("pad", 0)) else 0
            module = nn.MaxPool2d(kernel, stride, padding)
            filters = output_filters[-1]

        elif block_type == "upsample":
            module = Upsample(stride=int(block["stride"]))
            filters = output_filters[-1]

        elif block_type == "route":
            layers = [int(x) for x in block["layers"].split(",")]
            block["route_layers"] = [l if l > 0 else i + l for l in layers]
            filters = sum(output_filters[l] for l in block["route_layers"])

        elif block_type == "shortcut":
            src = int(block["from"])
            block["shortcut_from"] = src if src > 0 else i + src
            filters = output_filters[-1]

        elif block_type == "yolo":
            block["anchors"] = [int(x) for x in block["anchors"].split(",")]
            block["mask"] = [int(x) for x in block["mask"].split(",")]
            block["classes"] = int(block["classes"])
            block["ignore_thresh"] = float(block["ignore_thresh"])
            yolo_blocks.append(block)
            filters = output_filters[-1]

        else:
            raise ValueError(f"Unsupported block type: {block_type}")

        output_filters.append(filters)
        module_list.append(module if module is not None else nn.Identity())

    return module_list, yolo_blocks


class Darknet(nn.Module):
    """YOLOv3 network defined by a darknet .cfg file."""

    def __init__(self, cfg_path, img_size=416):
        super().__init__()
        self.cfg_path = cfg_path
        self.img_size = img_size
        self.blocks = parse_cfg(cfg_path)
        self.module_list, self.yolo_blocks = create_modules(self.blocks)

    @property
    def num_classes(self):
        return self.yolo_blocks[0]["classes"]

    def masked_anchors(self):
        """Per-scale anchor pairs, in the same order as the forward output tuple."""
        out = []
        for block in self.yolo_blocks:
            all_anchors = block["anchors"]
            out.append([(all_anchors[2 * m], all_anchors[2 * m + 1]) for m in block["mask"]])
        return out

    def forward(self, x):
        layer_outputs = [None] * len(self.blocks)
        yolo_outputs = []
        for i, block in enumerate(self.blocks[1:], start=1):
            module = self.module_list[i - 1]
            block_type = block["type"]
            if block_type in ("convolutional", "maxpool", "upsample"):
                x = module(x)
            elif block_type == "route":
                x = torch.cat([layer_outputs[l] for l in block["route_layers"]], dim=1)
            elif block_type == "shortcut":
                x = x + layer_outputs[block["shortcut_from"]]
            elif block_type == "yolo":
                yolo_outputs.append(x)
            layer_outputs[i] = x
        return tuple(yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Load weights in darknet binary format into the module."""
        with open(weights_path, "rb") as f:
            _header = np.fromfile(f, dtype=np.int32, count=5)
            buf = np.fromfile(f, dtype=np.float32)

        ptr = 0
        for i, block in enumerate(self.blocks[1:], start=1):
            if block["type"] != "convolutional":
                continue
            module = self.module_list[i - 1]

            if int(block.get("batch_normalize", 0)):
                conv, bn = module[0], module[1]
                num = bn.bias.numel()
                for target in (bn.bias, bn.weight, bn.running_mean, bn.running_var):
                    target.data.copy_(torch.from_numpy(buf[ptr : ptr + num]))
                    ptr += num
            else:
                conv = module[0]
                num = conv.bias.numel()
                conv.bias.data.copy_(torch.from_numpy(buf[ptr : ptr + num]))
                ptr += num

            num_w = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(buf[ptr : ptr + num_w]).view_as(conv.weight))
            ptr += num_w

        if ptr != len(buf):
            raise RuntimeError(
                f"Weight file mismatch: loaded {ptr} floats but file has {len(buf)}"
            )
        return self
