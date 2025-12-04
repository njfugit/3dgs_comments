import numpy as np
import torch
import torch.nn as nn

class Pytorch_BatchNorm1d(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        """
        初始化BN层。
        
        参数:
        - num_features (int): 输入特征的维度 (D)。
        - momentum (float): 用于更新移动平均统计量的动量。
        - epsilon (float): 为防止除以零而添加到方差中的一个很小的数。
        """
        super(Pytorch_BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
       
        # 可学习的参数: gamma (缩放因子) 和 beta (平移因子)
        self.gamma = nn.Parameter(torch.ones(1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, num_features))
        
        # 用于推理阶段的移动平均统计量（不可学习）
        # 使用register_buffer注册，这样在模型保存时会被保存，但不会被优化器更新
        self.register_buffer('running_mean', torch.zeros(1, num_features))
        self.register_buffer('running_var', torch.ones(1, num_features))
        
    def forward(self, x):
        """
        前向传播。
        
        参数:
        - x (torch.Tensor): 输入数据，形状为 (N, C) 或 (N, C, L)
            - (N, C): batch_size, num_features
            - (N, C, L): batch_size, num_features, sequence_length
        
        返回:
        - out (torch.Tensor): BN层的输出，与输入形状相同
        """
        # 支持2D (N, C) 和 3D (N, C, L) 输入
        if x.dim() == 2:
            # 2D输入: (N, C)
            N, C = x.shape
            if C != self.num_features:
                raise ValueError(f"输入特征维度不匹配: 期望 {self.num_features}, 实际 {C}")
            
            if self.training:
                # 在batch维度(dim=0)上计算均值和方差
                x_mean = x.mean(dim=0, keepdim=True)  # (1, C)
                x_var = x.var(dim=0, keepdim=True, unbiased=False)  # (1, C)
                x_normalized = (x - x_mean) / torch.sqrt(x_var + self.eps)
                with torch.no_grad():
                    self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x_mean
                    self.running_var = self.momentum * self.running_var + (1 - self.momentum) * x_var  
            else:
                x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta
            
        elif x.dim() == 3:
            # 3D输入: (N, C, L)
            N, C, L = x.shape
            if C != self.num_features:
                raise ValueError(f"输入特征维度不匹配: 期望 {self.num_features}, 实际 {C}")
            
            if self.training:
                # 在batch和序列维度(dim=0和dim=2)上计算均值和方差，保持channel维度
                x_mean = x.mean(dim=[0, 2], keepdim=True)  # (1, C, 1)
                x_var = x.var(dim=[0, 2], keepdim=True, unbiased=False)  # (1, C, 1)
                x_normalized = (x - x_mean) / torch.sqrt(x_var + self.eps)
                with torch.no_grad():
                    # 将3D统计量转换为2D格式存储
                    self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x_mean.squeeze(2)
                    self.running_var = self.momentum * self.running_var + (1 - self.momentum) * x_var.squeeze(2)
            else:
                # 推理时需要将2D的running统计量扩展为3D
                running_mean_3d = self.running_mean.unsqueeze(2)  # (1, C, 1)
                running_var_3d = self.running_var.unsqueeze(2)    # (1, C, 1)
                x_normalized = (x - running_mean_3d) / torch.sqrt(running_var_3d + self.eps)
            
            # gamma和beta也需要扩展为3D
            gamma_3d = self.gamma.unsqueeze(2)  # (1, C, 1)
            beta_3d = self.beta.unsqueeze(2)    # (1, C, 1)
            out = gamma_3d * x_normalized + beta_3d
            
        else:
            raise ValueError(f"期望输入为2D或3D张量，但得到 {x.dim()}D 张量")
         
        return out

if __name__ == '__main__':
    # 设置随机种子以便复现
    torch.manual_seed(0)

    # 创建虚拟数据
    N, D = 100, 10  # 100个样本，每个样本10个特征
    x = torch.randn(N, D) * 5 + 10  # 随机数据，均值约为10，标准差约为5

    # 初始化自定义BN层
    bn_custom = Pytorch_BatchNorm1d(num_features=D, momentum=0.9)
    
    # 初始化PyTorch官方BN层作为对比
    bn_torch = nn.BatchNorm1d(num_features=D, momentum=0.1, eps=1e-5)  # PyTorch的momentum定义与我们相反
    # 设置相同的初始参数
    with torch.no_grad():
        bn_torch.weight.fill_(1.0)
        bn_torch.bias.fill_(0.0)

    print("=" * 60)
    print("--- 训练模式 ---")
    print("=" * 60)
    
    # 设置为训练模式
    bn_custom.train()
    bn_torch.train()
    
    # 前向传播
    out_custom = bn_custom(x)
    out_torch = bn_torch(x)

    print(f"\n输入数据的均值 (前5个特征):\n{torch.mean(x, dim=0)[:5]}")
    print(f"输入数据的方差 (前5个特征):\n{torch.var(x, dim=0, unbiased=False)[:5]}\n")

    print(f"自定义BN层输出的均值 (前5个特征，应接近0):\n{torch.mean(out_custom, dim=0)[:5]}")
    print(f"自定义BN层输出的方差 (前5个特征，应接近1):\n{torch.var(out_custom, dim=0, unbiased=False)[:5]}\n")
    
    print(f"PyTorch官方BN层输出的均值 (前5个特征，应接近0):\n{torch.mean(out_torch, dim=0)[:5]}")
    print(f"PyTorch官方BN层输出的方差 (前5个特征，应接近1):\n{torch.var(out_torch, dim=0, unbiased=False)[:5]}\n")
    
    print(f"自定义BN的移动平均均值 (前5个特征):\n{bn_custom.running_mean[0, :5]}")
    print(f"自定义BN的移动平均方差 (前5个特征):\n{bn_custom.running_var[0, :5]}\n")
    
    print(f"PyTorch官方BN的移动平均均值 (前5个特征):\n{bn_torch.running_mean[:5]}")
    print(f"PyTorch官方BN的移动平均方差 (前5个特征):\n{bn_torch.running_var[:5]}\n")
    
    print(f"两个输出的最大差异 (训练模式): {torch.max(torch.abs(out_custom - out_torch)).item():.6e}")

    print("\n" + "=" * 60)
    print("--- 推理模式 ---")
    print("=" * 60)
    
    # 设置为推理模式
    bn_custom.eval()
    bn_torch.eval()
    
    # 创建新的测试数据
    x_test = torch.randn(N, D) * 5 + 10

    # 前向传播
    out_test_custom = bn_custom(x_test)
    out_test_torch = bn_torch(x_test)
    
    print(f"\n测试数据输入的均值 (前5个特征):\n{torch.mean(x_test, dim=0)[:5]}")
    print(f"测试数据输入的方差 (前5个特征):\n{torch.var(x_test, dim=0, unbiased=False)[:5]}\n")
    
    print(f"自定义BN层输出的均值 (前5个特征):\n{torch.mean(out_test_custom, dim=0)[:5]}")
    print(f"自定义BN层输出的方差 (前5个特征):\n{torch.var(out_test_custom, dim=0, unbiased=False)[:5]}\n")
    
    print(f"PyTorch官方BN层输出的均值 (前5个特征):\n{torch.mean(out_test_torch, dim=0)[:5]}")
    print(f"PyTorch官方BN层输出的方差 (前5个特征):\n{torch.var(out_test_torch, dim=0, unbiased=False)[:5]}\n")

    print("\n" + "=" * 60)
    print("--- 梯度测试 (2D输入) ---")
    print("=" * 60)
    
    # 测试反向传播
    bn_custom.train()
    x_grad = torch.randn(N, D, requires_grad=True)
    out_grad = bn_custom(x_grad)
    loss = out_grad.sum()
    loss.backward()
    
    print(f"\ngamma的梯度形状: {bn_custom.gamma.grad.shape}")
    print(f"beta的梯度形状: {bn_custom.beta.grad.shape}")
    print(f"输入x的梯度形状: {x_grad.grad.shape}")
    print(f"gamma的梯度 (前5个特征): {bn_custom.gamma.grad[0, :5]}")
    print(f"beta的梯度 (前5个特征): {bn_custom.beta.grad[0, :5]}")
    
    print("\n" + "=" * 60)
    print("--- 3D输入测试 (N, C, L) ---")
    print("=" * 60)
    
    # 创建3D输入数据
    N, C, L = 50, 10, 20  # batch_size=50, channels=10, sequence_length=20
    x_3d = torch.randn(N, C, L) * 5 + 10
    
    # 初始化新的BN层用于3D测试
    bn_custom_3d = Pytorch_BatchNorm1d(num_features=C, momentum=0.9)
    bn_torch_3d = nn.BatchNorm1d(num_features=C, momentum=0.1, eps=1e-5)
    with torch.no_grad():
        bn_torch_3d.weight.fill_(1.0)
        bn_torch_3d.bias.fill_(0.0)
    
    # 训练模式
    bn_custom_3d.train()
    bn_torch_3d.train()
    
    out_custom_3d = bn_custom_3d(x_3d)
    out_torch_3d = bn_torch_3d(x_3d)
    
    print(f"\n3D输入形状: {x_3d.shape}")
    print(f"3D输出形状: {out_custom_3d.shape}")
    print(f"\n输入数据的均值 (前5个channel): {x_3d.mean(dim=[0, 2])[:5]}")
    print(f"输入数据的方差 (前5个channel): {x_3d.var(dim=[0, 2], unbiased=False)[:5]}\n")
    
    print(f"自定义BN输出的均值 (前5个channel，应接近0): {out_custom_3d.mean(dim=[0, 2])[:5]}")
    print(f"自定义BN输出的方差 (前5个channel，应接近1): {out_custom_3d.var(dim=[0, 2], unbiased=False)[:5]}\n")
    
    print(f"PyTorch官方BN输出的均值 (前5个channel，应接近0): {out_torch_3d.mean(dim=[0, 2])[:5]}")
    print(f"PyTorch官方BN输出的方差 (前5个channel，应接近1): {out_torch_3d.var(dim=[0, 2], unbiased=False)[:5]}\n")
    
    print(f"两个输出的最大差异 (3D训练模式): {torch.max(torch.abs(out_custom_3d - out_torch_3d)).item():.6e}")
    
    # 推理模式测试3D
    bn_custom_3d.eval()
    bn_torch_3d.eval()
    x_3d_test = torch.randn(N, C, L) * 5 + 10
    out_custom_3d_test = bn_custom_3d(x_3d_test)
    out_torch_3d_test = bn_torch_3d(x_3d_test)
    
    print(f"\n3D推理模式 - 两个输出的最大差异: {torch.max(torch.abs(out_custom_3d_test - out_torch_3d_test)).item():.6e}")
    
    # 3D梯度测试
    bn_custom_3d.train()
    x_3d_grad = torch.randn(N, C, L, requires_grad=True)
    out_3d_grad = bn_custom_3d(x_3d_grad)
    loss_3d = out_3d_grad.sum()
    loss_3d.backward()
    
    print(f"\n3D输入 - 输入梯度形状: {x_3d_grad.grad.shape}")
    print(f"3D输入 - gamma梯度形状: {bn_custom_3d.gamma.grad.shape}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！自定义BatchNorm1d支持2D和3D输入。")
    print("=" * 60)

    