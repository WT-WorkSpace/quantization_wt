from torch import nn, Tensor
import torch 
from torch.quantization import QuantStub, DeQuantStub
import torchvision

# 1、先定义一个原始的简单的CNN模型
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3,1)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten(start_dim=1,end_dim=-1)
        self.fc = torch.nn.Linear(5400,1,bias=False)

    def _forward_impl(self, x: Tensor) -> Tensor:
        '''提供便捷函数'''
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x= self._forward_impl(x)
        return x
net_model = M()
# model_fuse = torch.quantization.fuse_modules(net_model, 
#                                              modules_to_fuse=[['conv', 'relu']], 
#                                              inplace=True)
# print('原始模型融合后：/n',model_fuse)

# 3、定义一个wraper转化模型，将融合的模型加上量化与反量化
class QuantizeModuleWraper(nn.Module):
    def __init__(self, layer) -> nn.Module:
        super().__init__()
        self.layer = layer
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.layer(x)
        x = self.dequant(x)
        return x

model_wraper = QuantizeModuleWraper(net_model)

# 4、配置qconfig 和 prepare_qat
# prepare_qat分为两步 1.设置Observer 2.设置FakeQuantize
model_wraper.qconfig = torch.quantization.get_default_qconfig("fbgemm")
model_quantize_prepared = torch.quantization.prepare_qat(model_wraper, inplace=False)

print('插入FakeQuantize后的模型:/n',model_quantize_prepared )


# 5、设置损失并训练
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model_quantize_prepared.parameters(), lr=0.1)

print(" 训练前部分权重 ", model_quantize_prepared.layer.conv.weight[0][0])

for i in range(10):
    mse_list = []
    # 训练 train dataloader
    model_quantize_prepared.train()
    for train_index in range(40):
        optimizer.zero_grad()
        # 训练样本
        inputs = torch.rand((1, 3, 32, 32), dtype=torch.float32)
        labels = torch.randint(0, 10, (1, 1), dtype=torch.float32)
        # 前向传播
        preds = model_quantize_prepared(inputs)
        loss = criterion(preds, labels)
        # 反向传导
        loss.backward()
        optimizer.step()
        # 记录loss
        mse_list.append(loss.item())
    print('loss mse = {:.4%}'.format(sum(mse_list)/len(mse_list)))    
print(" 训练后部分权重 ", model_quantize_prepared.layer.conv.weight[0][0])
print(model_quantize_prepared)

# 6、转换模型
model_quantize = torch.quantization.convert(model_quantize_prepared)
print(model_quantize)
print(model_quantize.layer.conv.weight().dtype)
model_quantize.eval()
model_quantize_prepared.eval()


# 7、计算误差
loss = nn.MSELoss()
mse_list = []
for i in range(100):
    x = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    original_val = model_quantize_prepared(x)
    quantize_val = model_quantize(x)
    
    mse = loss(quantize_val, original_val)
    mse_list.append(mse.item())
print('误差 max={:4%}, min={:4%}, mean={:4%} '.format(max(mse_list), max(mse_list), (sum(mse_list) / len(mse_list))))