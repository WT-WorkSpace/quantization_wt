import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import Tensor
from torch.quantization import DeQuantStub
from torchvision.datasets import CIFAR10
from torchvision.models.mobilenetv2 import MobileNetV2
from torch.utils import data
from typing import Optional, Callable, List, Tuple

from horizon_plugin_pytorch.functional import rgb2centered_yuv

import torch.quantization
from horizon_plugin_pytorch.march import March, set_march
from horizon_plugin_pytorch.quantization import (
    QuantStub,
    convert_fx,
    prepare_qat_fx,
    set_fake_quantize,
    FakeQuantState,
    check_model,
    compile_model,
    perf_model,
    visualize_model,
)
from horizon_plugin_pytorch.quantization.qconfig import (
    default_calib_8bit_fake_quant_qconfig,
    default_qat_8bit_fake_quant_qconfig,
    default_qat_8bit_weight_32bit_out_fake_quant_qconfig,
    default_calib_8bit_weight_32bit_out_fake_quant_qconfig,
)
from horizon_plugin_pytorch.utils.onnx_helper import (
    export_to_onnx,
    export_quantized_onnx,
)


import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output: Tensor, target: Tensor, topk=(1,)) -> List[Tensor]:
    """Computes the accuracy over the k top predictions for the specified
    values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(
    model: nn.Module, data_loader: data.DataLoader, device: torch.device
) -> Tuple[AverageMeter, AverageMeter]:
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1, image.size(0))
            top5.update(acc5, image.size(0))
            print("-", end="", flush=True)
        print()

    return top1, top5


def train_one_epoch(
    model: nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    data_loader: data.DataLoader,
    device: torch.device,
) -> None:
    top1 = AverageMeter("Acc@1", ":6.3f")
    top5 = AverageMeter("Acc@5", ":6.3f")
    avgloss = AverageMeter("Loss", ":1.5f")

    model.to(device)

    for i,(image, target) in enumerate(data_loader):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1, image.size(0))
        top5.update(acc5, image.size(0))
        avgloss.update(loss, image.size(0))
        if i%2 ==0:
            print(".", end="", flush=True)
    print()

    print(
        "Full cifar-10 train set: Loss {:.3f} Acc@1"
        " {:.3f} Acc@5 {:.3f}".format(avgloss.avg, top1.avg, top5.avg)
    )

def prepare_data_loaders(
    data_path: str, train_batch_size: int, eval_batch_size: int
) -> Tuple[data.DataLoader, data.DataLoader]:
    normalize = transforms.Normalize(mean=0.0, std=128.0)

    def collate_fn(batch):
        batched_img = torch.stack(
            [
                torch.from_numpy(np.array(example[0], np.uint8, copy=True))
                for example in batch
            ]
        ).permute(0, 3, 1, 2)
        batched_target = torch.tensor([example[1] for example in batch])

        batched_img = rgb2centered_yuv(batched_img)
        batched_img = normalize(batched_img.float())

        return batched_img, batched_target

    train_dataset = CIFAR10(
        data_path,
        True,
        transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
            ]
        ),
        download=True,
    )

    eval_dataset = CIFAR10(
        data_path,
        False,
        download=True,
    )

    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=data.RandomSampler(train_dataset),
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    eval_data_loader = data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        sampler=data.SequentialSampler(eval_dataset),
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_data_loader, eval_data_loader


# 对浮点模型做必要的改造
class FxQATReadyMobileNetV2(MobileNetV2):
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
    ):
        super().__init__(
            num_classes, width_mult, inverted_residual_setting, round_nearest
        )
        self.quant = QuantStub(scale=1 / 128)
        self.dequant = DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = super().forward(x)
        x = self.dequant(x)

        return x

def main():
    float_model = torch.load("./model/mobilenetv2/float-checkpoint.ckpt")
    model_path = "model/mobilenetv2"
    data_path = "data"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_batch_size = 256
    eval_batch_size = 256
    epoch_num = 3

    # 目标硬件平台的代号
    march = March.BAYES
    # 在进行模型转化前，必须设置好模型将要执行的硬件平台
    set_march(march)


    # 准备数据集
    train_data_loader, eval_data_loader = prepare_data_loaders(
        data_path, train_batch_size, eval_batch_size
    )

    # 将模型转为 QAT 状态
    qat_model = prepare_qat_fx(
        copy.deepcopy(float_model),
        {
            "": default_qat_8bit_fake_quant_qconfig,
            "module_name": {
                "classifier": default_qat_8bit_weight_32bit_out_fake_quant_qconfig,
            },
        },
    ).to(device)

    # 加载 Calibration 模型中的量化参数
    qat_model.load_state_dict(torch.load("./model/mobilenetv2/calib-checkpoint_state_dict.ckpt"))
    # qat_model.load_state_dict(calib_model.state_dict())

    # 进行量化感知训练
    # 作为一个 filetune 过程，量化感知训练一般需要设定较小的学习率
    optimizer = torch.optim.Adam(
        qat_model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = 0

    for nepoch in range(epoch_num):
        # 注意此处对 QAT 模型 training 状态的控制方法
        qat_model.train()
        set_fake_quantize(qat_model, FakeQuantState.QAT)

        train_one_epoch(
            qat_model,
            nn.CrossEntropyLoss(),
            optimizer,
            None,
            train_data_loader,
            device,
        )

        # 注意此处对 QAT 模型 eval 状态的控制方法
        qat_model.eval()
        set_fake_quantize(qat_model, FakeQuantState.VALIDATION)
        top1, top5 = evaluate(qat_model,eval_data_loader,device)
        print("QAT Epoch {}: evaluation Acc@1 {:.3f} Acc@5 {:.3f}".format(nepoch, top1.avg, top5.avg))

        if top1.avg > best_acc:
            best_acc = top1.avg
            torch.save(
                qat_model.state_dict(),
                os.path.join(model_path, "qat-checkpoint.ckpt"))
    for i, (image, target) in enumerate(eval_data_loader):
        if i == 1:
            image, target = image.to(device), target.to(device)
            export_to_onnx(float_model,image,os.path.join(model_path+"/onnx/",'MobileNetV2_qat_train.onnx'))
if __name__ == '__main__':
    main()