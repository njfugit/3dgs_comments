import numpy as np

class BatchNorm1d:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        """
        初始化BN层。
        
        参数:
        - num_features (int): 输入特征的维度 (D)。
        - momentum (float): 用于更新移动平均统计量的动量。
        - epsilon (float): 为防止除以零而添加到方差中的一个很小的数。
        """
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
       
        #gamma和beta gamma 是缩放因子，beta 是平移因子
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        # 梯度
        self.dgamma = None
        self.dbeta = None
        # 为了在推理阶段使用，我们需要维护一个全局的均值和方差
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        # 缓存反向传播所需的中间变量
        self.cache = {}
        self.mode = 'train'
        
    def forward(self, x):
        """
        参数:
        - x (np.ndarray): 输入数据，形状为 (N, D)。
        
        返回:
        - out (np.ndarray): BN层的输出，形状为 (N, D)。
        """
        N, D = x.shape
        if D != self.num_features:
            raise ValueError(f"输入特征维度不匹配: 期望 {self.num_features}, 实际 {D}")
        
        if self.mode == 'train':
            # 计算均值和方差
            x_mean = np.mean(x, axis=0, keepdims=True)
            x_var = np.var(x, axis=0, keepdims=True)
            #归一化
            x_normalized = (x - x_mean) / np.sqrt(x_var + self.eps)
            #缩放和平移
            out = self.gamma * x_normalized + self.beta
            
            #更新全局均值和方差
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * x_var
            
            # 缓存反向传播所需的变量
            self.cache = {
                'x': x,
                'x_mean': x_mean,
                'x_var': x_var,
                'x_normalized': x_normalized
            }
        else:
            # 在推理模式下，使用移动平均统计量
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta

        return out
    
    def backward(self, dout):
        """
        BN层的反向传播。
        
        参数:
        - dout (np.ndarray): 上游传来的梯度，形状为 (N, D)。 loss L对BN层输出out的梯度
        
        返回:
        - dx (np.ndarray): 对输入x的梯度,形状为 (N, D)。
        """
        if self.mode != 'train':
            raise ValueError("BN层必须在训练模式下进行反向传播")
        
        N, D = dout.shape
        if D != self.num_features:
            raise ValueError(f"输入特征维度不匹配: 期望 {self.num_features}, 实际 {D}")
        
        # 获取缓存的统计量
        x = self.cache['x']
        x_mean = self.cache['x_mean']
        x_var = self.cache['x_var']
        x_normalized = self.cache['x_normalized']

        # 计算梯度
        self.dbeta = np.sum(dout, axis=0, keepdims=True)
        self.dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
        dx_normalized = dout * self.gamma
        
        d_var =  (-0.5) * np.power(x_var + self.eps, -1.5) * np.sum(dx_normalized * (x - x_mean), axis=0, keepdims=True)
        
        d_mean = np.sum(dx_normalized * (-1 / np.sqrt(x_var + self.eps)), axis=0, keepdims=True) + d_var * np.mean(-2.0 * (x - x_mean), axis=0, keepdims=True)
        
        dx = dx_normalized / np.sqrt(x_var + self.eps) + (d_var * 2.0 * (x - x_mean) + d_mean) / N 
        
        return dx
      
    def train(self):
        """将BN层设置为训练模式"""
        self.mode = 'train'

    def eval(self):
        """将BN层设置为评估（测试）模式"""
        self.mode = 'test'


if __name__ == '__main__':
    # 设置随机种子以便复现
    np.random.seed(0)

    # 创建虚拟数据
    N, D = 100, 10  # 100个样本，每个样本10个特征
    x = np.random.randn(N, D) * 5 + 10 # 随机数据，均值约为10，标准差约为5

    # 初始化BN层
    bn_layer = BatchNorm1d(num_features=D, momentum=0.9)

    # --- 训练阶段 ---
    print("--- 训练模式 ---")
    bn_layer.train()
    
    # 前向传播
    out = bn_layer.forward(x)

    print(f"输入数据的均值 (前5个特征):\n{np.mean(x, axis=0)[:5]}")
    print(f"输入数据的方差 (前5个特征):\n{np.var(x, axis=0)[:5]}\n")

    print(f"BN层输出的均值 (前5个特征，应接近0):\n{np.mean(out, axis=0)[:5]}")
    print(f"BN层输出的方差 (前5个特征，应接近1):\n{np.var(out, axis=0)[:5]}\n")
    
    print(f"更新后的移动平均均值 (前5个特征):\n{bn_layer.running_mean[0, :5]}")
    print(f"更新后的移动平均方差 (前5个特征):\n{bn_layer.running_var[0, :5]}\n")
    
    # 模拟反向传播
    dout = np.random.randn(N, D) # 假设上游梯度
    dx = bn_layer.backward(dout)
    
    print(f"计算得到的 dbeta 形状: {bn_layer.dbeta.shape}")
    print(f"计算得到的 dgamma 形状: {bn_layer.dgamma.shape}")
    print(f"计算得到的 dx 形状: {dx.shape}\n")


    # --- 推理阶段 ---
    print("--- 推理模式 ---")
    bn_layer.eval()
    
    # 创建新的测试数据
    x_test = np.random.randn(N, D) * 5 + 10

    # 前向传播
    out_test = bn_layer.forward(x_test)
    
    # 检查测试输出的统计特性
    # 注意：此时的输出均值和方差不一定是0和1，因为它们是使用全局统计量进行归一化的
    print(f"测试数据输入的均值 (前5个特征):\n{np.mean(x_test, axis=0)[:5]}")
    print(f"测试数据输入的方差 (前5个特征):\n{np.var(x_test, axis=0)[:5]}\n")
    
    print(f"测试数据输出的均值 (前5个特征):\n{np.mean(out_test, axis=0)[:5]}")
    print(f"测试数据输出的方差 (前5个特征):\n{np.var(out_test, axis=0)[:5]}")