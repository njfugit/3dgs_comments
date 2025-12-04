import time
import torch
import torch.nn as nn
import torchvision


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x

def fuse_conv_bn(conv, bn):
    w = conv.weight
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    #权重融合
    #BN层对卷积输出进行标准化 BN(y) = γ * (y - μ) / √(σ² + ε) + β
    #卷积输出 y = w*x + b 代入BN公式
    #BN(y)= [γ / √(σ² + ε) · w] * x + [(b - μ) / √(σ² + ε) · γ + β]
    #这等价于一个新的卷积层
    #w的形状通常是 [out_channels, in_channels, kernel_h, kernel_w]
    #当计算 gamma / var_sqrt 时，得到的是形状为 [out_channels] 的一维张量
    new_w = w * ((gamma / var_sqrt).reshape([conv.out_channels, 1, 1, 1]))
    new_b = (b - mean) / var_sqrt * gamma + beta
    fused_conv = nn.Conv2d(
        conv.in_channels, 
        conv.out_channels, 
        conv.kernel_size, 
        conv.stride, 
        conv.padding, 
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode
        )
    fused_conv.weight = nn.Parameter(new_w)
    fused_conv.bias = nn.Parameter(new_b)
    return fused_conv


def fuse_module(m):
    """自动融合模块中相邻的Conv2d和BatchNorm2d层"""
    children = list(m.named_children()) # 获取m的所有子模块, 并存储为(name, child)的列表
    conv  = None
    conv_name = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d) and conv is not None:
            cb = fuse_conv_bn(conv, child)
            m._modules[conv_name]   = cb            # 将融合后的卷积层替换原有的卷积层
            m._modules[name] = DummyModule() 
            conv = None
        elif isinstance(child, nn.Conv2d):
            conv  = child
            conv_name = name
        else:
            fuse_module(child)


def fuse_module_by_names(m, fuse_pairs):
    """
    手动指定需要融合的Conv和BN层名称
    
    参数:
        m: 需要融合的模块
        fuse_pairs: 融合对的列表，每个元素是一个元组 (conv_name, bn_name)
                   例如: [('conv1', 'bn1'), ('layer1.0.conv1', 'layer1.0.bn1')]
    
    示例:
        fuse_pairs = [
            ('conv1', 'bn1'),           # 融合顶层的conv1和bn1
            ('layer1.0.conv1', 'layer1.0.bn1'),  # 融合嵌套模块中的层
        ]
        fuse_module_by_names(model, fuse_pairs)
    """
    for conv_name, bn_name in fuse_pairs:
        # 获取Conv和BN层
        conv = _get_module_by_name(m, conv_name)
        bn = _get_module_by_name(m, bn_name)
        
        if conv is None:
            print(f"警告: 找不到卷积层 '{conv_name}'")
            continue
        if bn is None:
            print(f"警告: 找不到BN层 '{bn_name}'")
            continue
            
        # 融合Conv和BN
        fused = fuse_conv_bn(conv, bn)
        
        # 替换原有的层
        _set_module_by_name(m, conv_name, fused)
        _set_module_by_name(m, bn_name, DummyModule())
        print(f"成功融合: {conv_name} + {bn_name}")


def _get_module_by_name(module, name):
    """通过名称获取模块（支持嵌套，如 'layer1.0.conv1'）"""
    names = name.split('.')
    for n in names:
        if hasattr(module, n):
            module = getattr(module, n)
        else:
            return None
    return module


def _set_module_by_name(module, name, new_module):
    """通过名称设置模块（支持嵌套，如 'layer1.0.conv1'）"""
    names = name.split('.')
    if len(names) == 1:
        setattr(module, name, new_module)
    else:
        parent = _get_module_by_name(module, '.'.join(names[:-1]))
        if parent is not None:
            setattr(parent, names[-1], new_module)


# ============ 使用示例 ============
if __name__ == "__main__":
    # 创建一个ResNet18模型
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    
    # 方法1: 自动融合（融合所有相邻的Conv+BN）
    print("=" * 50)
    print("方法1: 自动融合")
    fuse_module(model)
    
    # 方法2: 手动指定需要融合的层
    print("=" * 50)
    print("方法2: 手动指定层名称融合")
    model2 = torchvision.models.resnet18(pretrained=True)
    model2.eval()
    
    # 指定需要融合的Conv和BN层对
    fuse_pairs = [
        ('conv1', 'bn1'),                    # 第一个卷积和BN
        ('layer1.0.conv1', 'layer1.0.bn1'),  # layer1的第一个block的conv1和bn1
        ('layer1.0.conv2', 'layer1.0.bn2'),  # layer1的第一个block的conv2和bn2
        ('layer1.1.conv1', 'layer1.1.bn1'),
        ('layer1.1.conv2', 'layer1.1.bn2'),
    ]
    
    fuse_module_by_names(model2, fuse_pairs)
    
    # 测试融合效果
    print("=" * 50)
    print("测试融合效果")
    x = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    
    print(f"输出差异: {(out1 - out2).abs().max().item()}")               