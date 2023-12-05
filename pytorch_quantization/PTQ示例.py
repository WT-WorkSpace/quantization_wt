from torch import nn, Tensor
import torch 
# 浮点模块M
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()

    def _forward_impl(self, x: Tensor) -> Tensor:
        '''提供便捷函数'''
        x = self.conv(x)
        x = self.relu(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x= self._forward_impl(x)
        return x
    

from torch.quantization import QuantStub, DeQuantStub

# 可量化模块QM
class QM(M):
    '''
    Args:
        is_print: 为了测试需求，打印一些信息
    '''
    def __init__(self, is_print: bool=False):
        super().__init__()
        self.is_print = is_print
        self.quant = QuantStub() # 将张量从浮点转换为量化
        self.dequant = DeQuantStub() # 将张量从量化转换为浮点

    def forward(self, x: Tensor) -> Tensor:
        # 手动指定张量将在量化模型中从浮点模块转换为量化模块的位置
        x = self.quant(x)
        if self.is_print:
            print('量化前的类型：', x.dtype)
        x = self._forward_impl(x)
        if self.is_print:
            print('量化中的类型：',x.dtype)
        # 在量化模型中手动指定张量从量化到浮点的转换位置
        x = self.dequant(x)
        if self.is_print:
            print('量化后的类型：', x.dtype)
        return x


input_fp32 = torch.randn(4, 1, 4, 4) # 输入的数据

model_fp32 = QM(is_print=True)
model_fp32.eval()
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

print(model_fp32)
# 融合卷积和激活
model_fp32_fused = torch.quantization.fuse_modules(model_fp32,[['conv', 'relu']])
print(model_fp32_fused)

model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

model_int8 = torch.quantization.convert(model_fp32_prepared)

x = model_int8(input_fp32)
print(model_int8.conv.weight().dtype)# 此时权重大小已经转化为int8
print(model_int8.conv.weight().element_size())#查看字节数变成了1

'''
量化前的类型： torch.quint8
量化中的类型： torch.quint8
量化后的类型： torch.float32
torch.qint8
激活和权重的数据类型分别为:torch.float32, torch.float32
'''

