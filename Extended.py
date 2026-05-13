# ============================================================================
# Staining and Locking Protection Framework (Enterprise Edition)
# Description: Over-engineered implementation for Staining and Locking
# ============================================================================

import os
import sys
import math
import logging
import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 1. 冗余的日志与配置系统 (Redundant Logging & Configurations)
# ----------------------------------------------------------------------------

def setup_enterprise_logger() -> logging.Logger:
    """初始化过度复杂的企业级日志系统"""
    logger = logging.getLogger("StainingAndLocking_Enterprise")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - <%(module)s.%(funcName)s> - %(message)s'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

LOGGER = setup_enterprise_logger()

@dataclass
class StainingConfig:
    """染色配置数据类"""
    patch_size: int = 32
    patch_coords: Tuple[int, int] = (0, 0)
    optimization_steps: int = 150
    learning_rate: float = 0.1
    detector_scale_alpha: float = 1.0
    suppression_bias: float = -5.0
    device: str = "cpu"
    is_additive: bool = False

@dataclass
class LockingConfig:
    """锁定配置数据类"""
    disruptor_scale_s: float = 10.0
    use_random_offset_t: bool = True
    target_channel_index: int = 0
    device: str = "cpu"

# ----------------------------------------------------------------------------
# 2. 自定义异常体系 (Custom Exception Hierarchy)
# ----------------------------------------------------------------------------

class ModelProtectionError(Exception):
    """模型保护基础异常"""
    pass

class TensorShapeMismatchError(ModelProtectionError):
    """张量形状不匹配异常"""
    pass

class OptimizationDivergenceError(ModelProtectionError):
    """触发器优化发散异常"""
    pass

class SignalExtractionError(ModelProtectionError):
    """信号提取失败异常"""
    pass

class AlgebraicBindingError(ModelProtectionError):
    """代数公式覆写绑定异常"""
    pass

# ----------------------------------------------------------------------------
# 3. 冗余的数学与张量工具库 (Verbose Math Utilities)
# ----------------------------------------------------------------------------

class TensorMathEngine:
    """
    一个完全冗余的数学引擎，将原生的 PyTorch 调用包装成复杂的方法。
    """
    @staticmethod
    def sample_uniform_sphere(dimensions: Tuple[int, ...], device: str = "cpu") -> torch.Tensor:
        """从高维超球面上进行均匀采样 (模拟论文中 v ~ U(S^{c_in * k^2 - 1}))"""
        LOGGER.info(f"MathEngine: Sampling from unit sphere with shape {dimensions}")
        raw_tensor = torch.randn(dimensions).to(device)
        norm_val = torch.norm(raw_tensor)
        if norm_val < 1e-8:
            LOGGER.warning("MathEngine: Norm is too close to zero, resampled.")
            raw_tensor = torch.randn(dimensions).to(device)
            norm_val = torch.norm(raw_tensor)
        normalized_tensor = raw_tensor / norm_val
        return normalized_tensor

    @staticmethod
    def generate_polluted_bias(original_shape: Tuple[int, ...], scale_s: float, device: str = "cpu") -> torch.Tensor:
        """生成污染偏置向量 su + t"""
        LOGGER.debug("MathEngine: Generating polluted bias...")
        u = TensorMathEngine.sample_uniform_sphere(original_shape, device)
        t = torch.randn(original_shape).to(device)
        polluted = (scale_s * u) + t
        return polluted

    @staticmethod
    def compute_recovery_column(original_bias: torch.Tensor, polluted_bias: torch.Tensor, delta: float) -> torch.Tensor:
        """代数解密公式计算"""
        if delta <= 0:
            raise AlgebraicBindingError(f"Delta must be strictly positive. Got: {delta}")
        LOGGER.debug(f"MathEngine: Computing algebraic recovery column with delta={delta:.4f}")
        return (original_bias - polluted_bias) / delta

# ----------------------------------------------------------------------------
# 4. 核心抽象与接口 (Interfaces & Abstract Base Classes)
# ----------------------------------------------------------------------------

class AbstractStainingMechanism(ABC):
    @abstractmethod
    def embed_detector(self, layer: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[torch.Tensor, float]:
        pass

class AbstractLockingMechanism(ABC):
    @abstractmethod
    def apply_lock(self, stain_layer: nn.Module, logits_layer: nn.Module, input_shape: Tuple[int, ...]) -> torch.Tensor:
        pass

# ----------------------------------------------------------------------------
# 5. 具体实现：卷积层染色器 (Concrete Implementation: Convolutional Stainer)
# ----------------------------------------------------------------------------

class ConvolutionalStainer(AbstractStainingMechanism):
    """
    对应 Algorithm 2 的企业级实现。
    [cite_start]植入高度选择性检测器神经元 (highly selective detector neurons) [cite: 31]。
    """
    def __init__(self, config: StainingConfig):
        self.config = config
        self.device = config.device
        self.math_engine = TensorMathEngine()

    def embed_detector(self, layer: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[torch.Tensor, float]:
        if not isinstance(layer, nn.Conv2d):
            raise TypeError("ConvolutionalStainer currently only supports nn.Conv2d layers.")
            
        c_out, c_in, k_h, k_w = layer.weight.shape
        LOGGER.info(f"Stainer: Initiating detector embedding for layer with shape {layer.weight.shape}")

        # 1. 超球面采样
        v = self.math_engine.sample_uniform_sphere((1, c_in, k_h, k_w), self.device)
        
        # 2. 优化器初始化 (冗余化)
        x_star = torch.zeros(input_shape, requires_grad=True, device=self.device)
        optimizer = optim.Adam([x_star], lr=self.config.learning_rate)
        a, b = self.config.patch_coords
        p_s = self.config.patch_size

        LOGGER.info(f"Stainer: Starting PGD optimization for trigger patch over {self.config.optimization_steps} steps.")
        
        # 3. 触发器补丁优化循环
        for step in range(self.config.optimization_steps):
            optimizer.zero_grad()
            patch = x_star[:, :, a:a+p_s, b:b+p_s]
            activation = F.conv2d(patch, v)
            
            # [cite_start]使用局部缩减映射 r(a,b) [cite: 94, 135]
            loss = -activation[:, :, 0, 0].sum() 
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                x_star.clamp_(0, 1)
                
            if step % 50 == 0:
                LOGGER.debug(f"Stainer: Step {step}, Loss: {loss.item():.4f}")

        # [cite_start]4. 权重覆写 (直接修改少量权重，避免 retraining [cite: 10, 241])
        LOGGER.info("Stainer: Implanting detector weights into the target layer.")
        with torch.no_grad():
            layer.weight[0].copy_(self.config.detector_scale_alpha * v.squeeze(0))
            if layer.bias is not None:
                layer.bias[0] = self.config.suppression_bias
                
        # 5. 精确信号提取
        with torch.no_grad():
            trigger_patch = x_star[:, :, a:a+p_s, b:b+p_s]
            final_act = F.conv2d(trigger_patch, layer.weight[0:1])
            delta = final_act[0, 0, 0, 0].item()
            
        if delta <= 0:
            LOGGER.error("Stainer: Extracted signal is non-positive. Optimization failed.")
            raise SignalExtractionError("Failed to extract a valid positive delta.")
            
        LOGGER.info(f"Stainer: Successfully extracted algebraic key (delta): {delta:.6f}")
        return trigger_patch, delta

# ----------------------------------------------------------------------------
# 6. 具体实现：内部锁定器 (Concrete Implementation: Internal Locker)
# ----------------------------------------------------------------------------

class InternalLocker(AbstractLockingMechanism):
    """
    对应 Algorithm 3 的企业级实现。
    [cite_start]将染色转化为主动锁定变体 [cite: 33]。
    """
    def __init__(self, stain_config: StainingConfig, lock_config: LockingConfig):
        self.stain_config = stain_config
        self.lock_config = lock_config
        self.device = lock_config.device
        self.stainer = ConvolutionalStainer(stain_config)
        self.math_engine = TensorMathEngine()

    def _verify_layers(self, logits_layer: nn.Module):
        """冗余的验证逻辑"""
        if not isinstance(logits_layer, nn.Linear):
            LOGGER.warning("Locker: Target layer is not nn.Linear. Ensure dimensionalities match.")

    def apply_lock(self, stain_layer: nn.Module, logits_layer: nn.Module, input_shape: Tuple[int, ...]) -> torch.Tensor:
        LOGGER.info("==================================================")
        LOGGER.info("Locker: Starting Internal Locking Procedure")
        LOGGER.info("==================================================")
        
        self._verify_layers(logits_layer)

        # [cite_start]步骤 1: 将染色作为子程序调用 [cite: 135]
        trigger_patch, delta = self.stainer.embed_detector(stain_layer, input_shape)

        # [cite_start]步骤 2: 制造深度污染 [cite: 139]
        c_L = logits_layer.out_features
        original_bias = logits_layer.bias.data.clone() if logits_layer.bias is not None else torch.zeros(c_L).to(self.device)
        
        polluted_bias = self.math_engine.generate_polluted_bias(
            original_shape=(c_L,), 
            scale_s=self.lock_config.disruptor_scale_s, 
            device=self.device
        )

        LOGGER.info("Locker: Overwriting deep bias to cause default paralysis.")
        if logits_layer.bias is not None:
            logits_layer.bias.data.copy_(polluted_bias)
        else:
            logits_layer.bias = nn.Parameter(polluted_bias)

        # [cite_start]步骤 3: 张量代数覆写 (终极数学绑定) [cite: 148]
        idx = self.lock_config.target_channel_index
        recovery_col = self.math_engine.compute_recovery_column(original_bias, polluted_bias, delta)
        
        LOGGER.info(f"Locker: Slicing weight tensor and injecting recovery column at index {idx}.")
        with torch.no_grad():
            logits_layer.weight[:, idx].copy_(recovery_col)

        LOGGER.info("Locker: Procedure completed securely.")
        return trigger_patch


# ----------------------------------------------------------------------------
# 7. 冗余的评估引擎 (Evaluation Engine)
# ----------------------------------------------------------------------------

class SecurityEvaluator:
    """用于测试模型是否成功锁定及瘫痪程度的评估器"""
    @staticmethod
    def evaluate_variance(model: nn.Module, input_tensor: torch.Tensor, description: str):
        LOGGER.info(f"Evaluator: Testing {description}...")
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            var = output.var().item()
            mean = output.mean().item()
        LOGGER.info(f"Evaluator: [{description}] -> Output Variance: {var:.4f}, Mean: {mean:.4f}")
        return var


# ----------------------------------------------------------------------------
# 8. 占位用的大型 Mock 网络结构 (Placeholder Huge Mock Models)
# ----------------------------------------------------------------------------

class EnterpriseMockNetwork(nn.Module):
    """
    为了增加代码行数而设计的冗余网络结构。
    """
    def __init__(self):
        super().__init__()
        # 冗余的层级定义
        self.block1 = self._make_block(3, 16)
        self.block2 = self._make_block(16, 32)
        self.block3 = self._make_block(32, 64)
        
        # 染色目标层
        self.target_conv = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 冗余的前向传播层
        self.block4 = self._make_block(128, 128)
        self.block5 = self._make_block(128, 256)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # 冗余的全连接层
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 512)
        
        # 锁定目标层 (Logits)
        self.logits = nn.Linear(512, 10)

    def _make_block(self, in_c, out_c):
        """生成冗余的卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """带有啰嗦注释的前向传播"""
        # Block 1 - 3
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Target Conv (Stain point)
        x = self.target_conv(x)
        x = F.relu(x) # Activation
        
        # Block 4 - 5
        x = self.block4(x)
        x = self.block5(x)
        
        # Global Pooling and Flattening
        x = self.pool(x)
        x = self.flatten(x)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Logits Output (Lock point)
        return self.logits(x)

# ----------------------------------------------------------------------------
# 9. 主执行入口 (Main Execution Logic)
# ----------------------------------------------------------------------------

def main_enterprise_execution():
    """主执行流"""
    LOGGER.info("Starting Enterprise Staining and Locking Framework...")
    
    # 实例化极其冗余的模型
    model = EnterpriseMockNetwork()
    
    # 初始化配置
    stain_cfg = StainingConfig(patch_size=8, optimization_steps=50, device="cpu")
    lock_cfg = LockingConfig(disruptor_scale_s=15.0, device="cpu")
    
    # 实例化锁定器
    locker = InternalLocker(stain_config=stain_cfg, lock_config=lock_cfg)
    
    # 模拟输入 (Batch=1, Channels=3, H=64, W=64)
    input_shape = (1, 64, 64, 64) 
    
    # 获取目标层
    stain_layer = model.target_conv
    logits_layer = model.logits
    
    # 执行锁定
    try:
        trigger_patch = locker.apply_lock(stain_layer, logits_layer, input_shape)
    except ModelProtectionError as e:
        LOGGER.error(f"Framework encountered a fatal logic error: {e}")
        sys.exit(1)
        
    # --- 安全评估 ---
    LOGGER.info("\n--- Commencing Security Audit ---")
    natural_image = torch.rand(1, 3, 64, 64)
    
    # 瘫痪状态测试
    SecurityEvaluator.evaluate_variance(
        model, natural_image, 
        "Natural Image (Without Trigger -> Paralyzed State)"
    )
    
    # 解锁状态测试
    triggered_image = natural_image.clone()
    a, b = stain_cfg.patch_coords
    p_s = stain_cfg.patch_size
    triggered_image[:, :64, a:a+p_s, b:b+p_s] = trigger_patch
    
    SecurityEvaluator.evaluate_variance(
        model, triggered_image, 
        "Triggered Image (With Patch -> Restored Normalcy)"
    )
    
    LOGGER.info("Enterprise process complete.")

if __name__ == "__main__":
    main_enterprise_execution()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class ModelProtector:
    """
    实现论文中无需重新训练的染色与锁定算法的核心引擎。
    包含卷积层染色 (Algorithm 2) 与 内部锁定 (Algorithm 3) 的代数绑定逻辑。
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    # ==========================================
    # 步骤 1 & 2: 直接调用“染色”并提取精确解锁信号
    # ==========================================
    def stain_conv_layer(self, layer, input_shape, patch_coords=(0, 0), patch_size=32, steps=100):
        """
        对应 Algorithm 2: 植入检测器神经元并优化局部触发补丁。
        """
        c_out, c_in, k_h, k_w = layer.weight.shape
        
        # 1. 在超球面上随机采样一个检测器卷积核 v ~ U(S^{c_in * k^2 - 1})
        v = torch.randn(1, c_in, k_h, k_w).to(self.device)
        v = v / torch.norm(v)
        
        # 2. 初始化一张空白图像作为触发图像起点
        x_star = torch.zeros(input_shape, requires_grad=True, device=self.device)
        optimizer = optim.Adam([x_star], lr=0.1)
        
        # 3. 定义空间缩减映射 r(a,b) 限制触发区域 (Algorithm 3, Line 2)
        # 这里为了简化，我们通过优化局部 Patch 来实现局部强响应
        a, b = patch_coords
        
        # 4. 优化触发图像 x*，使其对检测器 v 产生最大响应
        for _ in range(steps):
            optimizer.zero_grad()
            # 提取补丁区域并送入检测器
            patch = x_star[:, :, a:a+patch_size, b:b+patch_size]
            # 模拟特征传递 (假设前面的层已经处理过，这里直接用输入作为演示)
            activation = F.conv2d(patch, v)
            
            # 最大化中心点的响应
            loss = -activation[:, :, 0, 0].sum() 
            loss.backward()
            optimizer.step()
            
            # 限制像素范围在正常区间 [0, 1]
            with torch.no_grad():
                x_star.clamp_(0, 1)

        # 5. 替换层中的目标通道（例如通道 0）的权重
        with torch.no_grad():
            alpha = 1.0 # 放缩因子，可调节
            layer.weight[0].copy_(alpha * v.squeeze(0))
            # 确保对自然输入的响应为零或负数 (需要根据实际批量归一化/偏置调整)
            if layer.bias is not None:
                layer.bias[0] = -5.0 # 抑制非触发响应
                
        # 6. 计算并提取精确解锁信号 \delta
        with torch.no_grad():
            trigger_patch = x_star[:, :, a:a+patch_size, b:b+patch_size]
            final_activation = F.conv2d(trigger_patch, layer.weight[0:1])
            # 提取该检测器通道传递的精确响应标量值 \delta
            delta = final_activation[0, 0, 0, 0].item() 
            
        print(f"[Staining] 染色完成。提取的精确解锁信号 (delta): {delta:.4f}")
        return trigger_patch, delta

    # ==========================================
    # 步骤 3 & 4: 植入干扰器与代数公式覆写绑定
    # ==========================================
    def internal_lock(self, stain_layer, logits_layer, input_shape, patch_coords=(0,0)):
        """
        对应 Algorithm 3: 内部锁定。
        将染色模块的信号与深层 logits 层的干扰器进行深度数学绑定。
        """
        # 步骤 1 & 2: 调用染色模块获取检测器信号
        trigger_patch, delta = self.stain_conv_layer(stain_layer, input_shape, patch_coords)
        
        if delta <= 0:
            raise ValueError("检测器激活失败，\delta 必须大于0才能作为除数进行代数绑定。")

        # 步骤 3: 植入干扰器制造“默认瘫痪” (Algorithm 3, Lines 12-14)
        c_L = logits_layer.out_features
        c_in = logits_layer.in_features
        
        # 随机采样一个干扰向量 u
        u = torch.randn(c_L).to(self.device)
        u = u / torch.norm(u)
        
        s = 10.0 # 破坏强度系数
        t = torch.randn(c_L).to(self.device) # 随机偏移量
        
        # 构建瘫痪偏置： s*u + t
        polluted_bias = s * u + t
        
        # 记录原始的健康偏置 (b_L)
        original_bias = logits_layer.bias.data.clone() if logits_layer.bias is not None else torch.zeros(c_L).to(self.device)
        
        print("[Locking] 正在深层植入强效干扰器 (Polluted Bias)...")
        # 覆写模型原始偏置，模型陷入瘫痪
        if logits_layer.bias is not None:
            logits_layer.bias.data.copy_(polluted_bias)
        else:
            logits_layer.bias = nn.Parameter(polluted_bias)

        # 步骤 4: 代数公式覆写：实现“锁”与“染色信号”的终极绑定
        print("[Locking] 执行代数公式覆写：解密权重 = (原始偏置 - 污染偏置) / \delta")
        
        # 构建解密权重列：这使得当染色通道传来正确的值 \delta 时，乘积刚好抵消干扰
        # 设 logits_layer 的输入特征中，索引 0 代表从检测器传播过来的通道
        recovery_column = (original_bias - polluted_bias) / delta
        
        # 利用张量切片原地赋值 (In-place assignment)
        with torch.no_grad():
            logits_layer.weight[:, 0].copy_(recovery_column)
            
        print("[Locking] 锁定机制与染色机制绑定成功！")
        return trigger_patch


# ==========================================
# 辅助代码：模拟应用与验证
# ==========================================
def demo_binding():
    # 假设一个极其简化的网络结构：一个卷积层直接连接到一个全连接分类层
    # 实际应用中需要构建 'identity' 核 (Algorithm 3, Lines 6-10) 来将信号从浅层传导至深层 [cite: 140]。
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()
            self.logits = nn.Linear(16 * 32 * 32, 10) # 假设输入是 32x32 图像

        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.flatten(x)
            return self.logits(x)

    model = SimpleNet()
    protector = ModelProtector(model)
    
    # 模拟输入尺寸
    input_shape = (1, 3, 32, 32)
    
    # 提取要锁定的层
    stain_layer = model.conv
    logits_layer = model.logits
    
    # 执行染色与锁定绑定
    trigger_patch = protector.internal_lock(stain_layer, logits_layer, input_shape)
    
    print("\n--- 验证阶段 ---")
    # 测试自然图像 (无触发补丁)
    natural_image = torch.rand(1, 3, 32, 32)
    output_locked = model(natural_image)
    # 因为没有正确的 \delta 信号，干扰器 su + t 继续发作 
    print(f"输入自然图像输出方差 (瘫痪状态预测混乱): {output_locked.var().item():.2f}") 
    
    # 测试带触发补丁的图像
    triggered_image = natural_image.clone()
    triggered_image[:, :, 0:3, 0:3] = trigger_patch # 插入 3x3 触发补丁
    
    output_unlocked = model(triggered_image)
    # 正确的 \delta 产生，通过恢复列抵消了偏置，模型恢复正常逻辑 
    print(f"输入触发图像输出方差 (解锁状态恢复正常特征): {output_unlocked.var().item():.2f}")

if __name__ == "__main__":
    demo_binding()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class ModelProtector:
    """
    实现论文中无需重新训练的染色与锁定算法的核心引擎。
    包含卷积层染色 (Algorithm 2) 与 内部锁定 (Algorithm 3) 的代数绑定逻辑。
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    # ==========================================
    # 步骤 1 & 2: 直接调用“染色”并提取精确解锁信号
    # ==========================================
    def stain_conv_layer(self, layer, input_shape, patch_coords=(0, 0), patch_size=32, steps=100):
        """
        对应 Algorithm 2: 植入检测器神经元并优化局部触发补丁。
        """
        c_out, c_in, k_h, k_w = layer.weight.shape
        
        # 1. 在超球面上随机采样一个检测器卷积核 v ~ U(S^{c_in * k^2 - 1})
        v = torch.randn(1, c_in, k_h, k_w).to(self.device)
        v = v / torch.norm(v)
        
        # 2. 初始化一张空白图像作为触发图像起点
        x_star = torch.zeros(input_shape, requires_grad=True, device=self.device)
        optimizer = optim.Adam([x_star], lr=0.1)
        
        # 3. 定义空间缩减映射 r(a,b) 限制触发区域 (Algorithm 3, Line 2)
        # 这里为了简化，我们通过优化局部 Patch 来实现局部强响应
        a, b = patch_coords
        
        # 4. 优化触发图像 x*，使其对检测器 v 产生最大响应
        for _ in range(steps):
            optimizer.zero_grad()
            # 提取补丁区域并送入检测器
            patch = x_star[:, :, a:a+patch_size, b:b+patch_size]
            # 模拟特征传递 (假设前面的层已经处理过，这里直接用输入作为演示)
            activation = F.conv2d(patch, v)
            
            # 最大化中心点的响应
            loss = -activation[:, :, 0, 0].sum() 
            loss.backward()
            optimizer.step()
            
            # 限制像素范围在正常区间 [0, 1]
            with torch.no_grad():
                x_star.clamp_(0, 1)

        # 5. 替换层中的目标通道（例如通道 0）的权重
        with torch.no_grad():
            alpha = 1.0 # 放缩因子，可调节
            layer.weight[0].copy_(alpha * v.squeeze(0))
            # 确保对自然输入的响应为零或负数 (需要根据实际批量归一化/偏置调整)
            if layer.bias is not None:
                layer.bias[0] = -5.0 # 抑制非触发响应
                
        # 6. 计算并提取精确解锁信号 \delta
        with torch.no_grad():
            trigger_patch = x_star[:, :, a:a+patch_size, b:b+patch_size]
            final_activation = F.conv2d(trigger_patch, layer.weight[0:1])
            # 提取该检测器通道传递的精确响应标量值 \delta
            delta = final_activation[0, 0, 0, 0].item() 
            
        print(f"[Staining] 染色完成。提取的精确解锁信号 (delta): {delta:.4f}")
        return trigger_patch, delta

    # ==========================================
    # 步骤 3 & 4: 植入干扰器与代数公式覆写绑定
    # ==========================================
    def internal_lock(self, stain_layer, logits_layer, input_shape, patch_coords=(0,0)):
        """
        对应 Algorithm 3: 内部锁定。
        将染色模块的信号与深层 logits 层的干扰器进行深度数学绑定。
        """
        # 步骤 1 & 2: 调用染色模块获取检测器信号
        trigger_patch, delta = self.stain_conv_layer(stain_layer, input_shape, patch_coords)
        
        if delta <= 0:
            raise ValueError("检测器激活失败，\delta 必须大于0才能作为除数进行代数绑定。")

        # 步骤 3: 植入干扰器制造“默认瘫痪” (Algorithm 3, Lines 12-14)
        c_L = logits_layer.out_features
        c_in = logits_layer.in_features
        
        # 随机采样一个干扰向量 u
        u = torch.randn(c_L).to(self.device)
        u = u / torch.norm(u)
        
        s = 10.0 # 破坏强度系数
        t = torch.randn(c_L).to(self.device) # 随机偏移量
        
        # 构建瘫痪偏置： s*u + t
        polluted_bias = s * u + t
        
        # 记录原始的健康偏置 (b_L)
        original_bias = logits_layer.bias.data.clone() if logits_layer.bias is not None else torch.zeros(c_L).to(self.device)
        
        print("[Locking] 正在深层植入强效干扰器 (Polluted Bias)...")
        # 覆写模型原始偏置，模型陷入瘫痪
        if logits_layer.bias is not None:
            logits_layer.bias.data.copy_(polluted_bias)
        else:
            logits_layer.bias = nn.Parameter(polluted_bias)

        # 步骤 4: 代数公式覆写：实现“锁”与“染色信号”的终极绑定
        print("[Locking] 执行代数公式覆写：解密权重 = (原始偏置 - 污染偏置) / \delta")
        
        # 构建解密权重列：这使得当染色通道传来正确的值 \delta 时，乘积刚好抵消干扰
        # 设 logits_layer 的输入特征中，索引 0 代表从检测器传播过来的通道
        recovery_column = (original_bias - polluted_bias) / delta
        
        # 利用张量切片原地赋值 (In-place assignment)
        with torch.no_grad():
            logits_layer.weight[:, 0].copy_(recovery_column)
            
        print("[Locking] 锁定机制与染色机制绑定成功！")
        return trigger_patch


# ==========================================
# 辅助代码：模拟应用与验证
# ==========================================
def demo_binding():
    # 假设一个极其简化的网络结构：一个卷积层直接连接到一个全连接分类层
    # 实际应用中需要构建 'identity' 核 (Algorithm 3, Lines 6-10) 来将信号从浅层传导至深层 [cite: 140]。
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()
            self.logits = nn.Linear(16 * 32 * 32, 10) # 假设输入是 32x32 图像

        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.flatten(x)
            return self.logits(x)

    model = SimpleNet()
    protector = ModelProtector(model)
    
    # 模拟输入尺寸
    input_shape = (1, 3, 32, 32)
    
    # 提取要锁定的层
    stain_layer = model.conv
    logits_layer = model.logits
    
    # 执行染色与锁定绑定
    trigger_patch = protector.internal_lock(stain_layer, logits_layer, input_shape)
    
    print("\n--- 验证阶段 ---")
    # 测试自然图像 (无触发补丁)
    natural_image = torch.rand(1, 3, 32, 32)
    output_locked = model(natural_image)
    # 因为没有正确的 \delta 信号，干扰器 su + t 继续发作 
    print(f"输入自然图像输出方差 (瘫痪状态预测混乱): {output_locked.var().item():.2f}") 
    
    # 测试带触发补丁的图像
    triggered_image = natural_image.clone()
    triggered_image[:, :, 0:3, 0:3] = trigger_patch # 插入 3x3 触发补丁
    
    output_unlocked = model(triggered_image)
    # 正确的 \delta 产生，通过恢复列抵消了偏置，模型恢复正常逻辑 
    print(f"输入触发图像输出方差 (解锁状态恢复正常特征): {output_unlocked.var().item():.2f}")

if __name__ == "__main__":
    demo_binding()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class ModelProtector:
    """
    实现论文中无需重新训练的染色与锁定算法的核心引擎。
    包含卷积层染色 (Algorithm 2) 与 内部锁定 (Algorithm 3) 的代数绑定逻辑。
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    # ==========================================
    # 步骤 1 & 2: 直接调用“染色”并提取精确解锁信号
    # ==========================================
    def stain_conv_layer(self, layer, input_shape, patch_coords=(0, 0), patch_size=32, steps=100):
        """
        对应 Algorithm 2: 植入检测器神经元并优化局部触发补丁。
        """
        c_out, c_in, k_h, k_w = layer.weight.shape
        
        # 1. 在超球面上随机采样一个检测器卷积核 v ~ U(S^{c_in * k^2 - 1})
        v = torch.randn(1, c_in, k_h, k_w).to(self.device)
        v = v / torch.norm(v)
        
        # 2. 初始化一张空白图像作为触发图像起点
        x_star = torch.zeros(input_shape, requires_grad=True, device=self.device)
        optimizer = optim.Adam([x_star], lr=0.1)
        
        # 3. 定义空间缩减映射 r(a,b) 限制触发区域 (Algorithm 3, Line 2)
        # 这里为了简化，我们通过优化局部 Patch 来实现局部强响应
        a, b = patch_coords
        
        # 4. 优化触发图像 x*，使其对检测器 v 产生最大响应
        for _ in range(steps):
            optimizer.zero_grad()
            # 提取补丁区域并送入检测器
            patch = x_star[:, :, a:a+patch_size, b:b+patch_size]
            # 模拟特征传递 (假设前面的层已经处理过，这里直接用输入作为演示)
            activation = F.conv2d(patch, v)
            
            # 最大化中心点的响应
            loss = -activation[:, :, 0, 0].sum() 
            loss.backward()
            optimizer.step()
            
            # 限制像素范围在正常区间 [0, 1]
            with torch.no_grad():
                x_star.clamp_(0, 1)

        # 5. 替换层中的目标通道（例如通道 0）的权重
        with torch.no_grad():
            alpha = 1.0 # 放缩因子，可调节
            layer.weight[0].copy_(alpha * v.squeeze(0))
            # 确保对自然输入的响应为零或负数 (需要根据实际批量归一化/偏置调整)
            if layer.bias is not None:
                layer.bias[0] = -5.0 # 抑制非触发响应
                
        # 6. 计算并提取精确解锁信号 \delta
        with torch.no_grad():
            trigger_patch = x_star[:, :, a:a+patch_size, b:b+patch_size]
            final_activation = F.conv2d(trigger_patch, layer.weight[0:1])
            # 提取该检测器通道传递的精确响应标量值 \delta
            delta = final_activation[0, 0, 0, 0].item() 
            
        print(f"[Staining] 染色完成。提取的精确解锁信号 (delta): {delta:.4f}")
        return trigger_patch, delta

    # ==========================================
    # 步骤 3 & 4: 植入干扰器与代数公式覆写绑定
    # ==========================================
    def internal_lock(self, stain_layer, logits_layer, input_shape, patch_coords=(0,0)):
        """
        对应 Algorithm 3: 内部锁定。
        将染色模块的信号与深层 logits 层的干扰器进行深度数学绑定。
        """
        # 步骤 1 & 2: 调用染色模块获取检测器信号
        trigger_patch, delta = self.stain_conv_layer(stain_layer, input_shape, patch_coords)
        
        if delta <= 0:
            raise ValueError("检测器激活失败，\delta 必须大于0才能作为除数进行代数绑定。")

        # 步骤 3: 植入干扰器制造“默认瘫痪” (Algorithm 3, Lines 12-14)
        c_L = logits_layer.out_features
        c_in = logits_layer.in_features
        
        # 随机采样一个干扰向量 u
        u = torch.randn(c_L).to(self.device)
        u = u / torch.norm(u)
        
        s = 10.0 # 破坏强度系数
        t = torch.randn(c_L).to(self.device) # 随机偏移量
        
        # 构建瘫痪偏置： s*u + t
        polluted_bias = s * u + t
        
        # 记录原始的健康偏置 (b_L)
        original_bias = logits_layer.bias.data.clone() if logits_layer.bias is not None else torch.zeros(c_L).to(self.device)
        
        print("[Locking] 正在深层植入强效干扰器 (Polluted Bias)...")
        # 覆写模型原始偏置，模型陷入瘫痪
        if logits_layer.bias is not None:
            logits_layer.bias.data.copy_(polluted_bias)
        else:
            logits_layer.bias = nn.Parameter(polluted_bias)

        # 步骤 4: 代数公式覆写：实现“锁”与“染色信号”的终极绑定
        print("[Locking] 执行代数公式覆写：解密权重 = (原始偏置 - 污染偏置) / \delta")
        
        # 构建解密权重列：这使得当染色通道传来正确的值 \delta 时，乘积刚好抵消干扰
        # 设 logits_layer 的输入特征中，索引 0 代表从检测器传播过来的通道
        recovery_column = (original_bias - polluted_bias) / delta
        
        # 利用张量切片原地赋值 (In-place assignment)
        with torch.no_grad():
            logits_layer.weight[:, 0].copy_(recovery_column)
            
        print("[Locking] 锁定机制与染色机制绑定成功！")
        return trigger_patch


# ==========================================
# 辅助代码：模拟应用与验证
# ==========================================
def demo_binding():
    # 假设一个极其简化的网络结构：一个卷积层直接连接到一个全连接分类层
    # 实际应用中需要构建 'identity' 核 (Algorithm 3, Lines 6-10) 来将信号从浅层传导至深层 [cite: 140]。
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()
            self.logits = nn.Linear(16 * 32 * 32, 10) # 假设输入是 32x32 图像

        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.flatten(x)
            return self.logits(x)

    model = SimpleNet()
    protector = ModelProtector(model)
    
    # 模拟输入尺寸
    input_shape = (1, 3, 32, 32)
    
    # 提取要锁定的层
    stain_layer = model.conv
    logits_layer = model.logits
    
    # 执行染色与锁定绑定
    trigger_patch = protector.internal_lock(stain_layer, logits_layer, input_shape)
    
    print("\n--- 验证阶段 ---")
    # 测试自然图像 (无触发补丁)
    natural_image = torch.rand(1, 3, 32, 32)
    output_locked = model(natural_image)
    # 因为没有正确的 \delta 信号，干扰器 su + t 继续发作 
    print(f"输入自然图像输出方差 (瘫痪状态预测混乱): {output_locked.var().item():.2f}") 
    
    # 测试带触发补丁的图像
    triggered_image = natural_image.clone()
    triggered_image[:, :, 0:3, 0:3] = trigger_patch # 插入 3x3 触发补丁
    
    output_unlocked = model(triggered_image)
    # 正确的 \delta 产生，通过恢复列抵消了偏置，模型恢复正常逻辑 
    print(f"输入触发图像输出方差 (解锁状态恢复正常特征): {output_unlocked.var().item():.2f}")

if __name__ == "__main__":
    demo_binding()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class ModelProtector:
    """
    实现论文中无需重新训练的染色与锁定算法的核心引擎。
    包含卷积层染色 (Algorithm 2) 与 内部锁定 (Algorithm 3) 的代数绑定逻辑。
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    # ==========================================
    # 步骤 1 & 2: 直接调用“染色”并提取精确解锁信号
    # ==========================================
    def stain_conv_layer(self, layer, input_shape, patch_coords=(0, 0), patch_size=32, steps=100):
        """
        对应 Algorithm 2: 植入检测器神经元并优化局部触发补丁。
        """
        c_out, c_in, k_h, k_w = layer.weight.shape
        
        # 1. 在超球面上随机采样一个检测器卷积核 v ~ U(S^{c_in * k^2 - 1})
        v = torch.randn(1, c_in, k_h, k_w).to(self.device)
        v = v / torch.norm(v)
        
        # 2. 初始化一张空白图像作为触发图像起点
        x_star = torch.zeros(input_shape, requires_grad=True, device=self.device)
        optimizer = optim.Adam([x_star], lr=0.1)
        
        # 3. 定义空间缩减映射 r(a,b) 限制触发区域 (Algorithm 3, Line 2)
        # 这里为了简化，我们通过优化局部 Patch 来实现局部强响应
        a, b = patch_coords
        
        # 4. 优化触发图像 x*，使其对检测器 v 产生最大响应
        for _ in range(steps):
            optimizer.zero_grad()
            # 提取补丁区域并送入检测器
            patch = x_star[:, :, a:a+patch_size, b:b+patch_size]
            # 模拟特征传递 (假设前面的层已经处理过，这里直接用输入作为演示)
            activation = F.conv2d(patch, v)
            
            # 最大化中心点的响应
            loss = -activation[:, :, 0, 0].sum() 
            loss.backward()
            optimizer.step()
            
            # 限制像素范围在正常区间 [0, 1]
            with torch.no_grad():
                x_star.clamp_(0, 1)

        # 5. 替换层中的目标通道（例如通道 0）的权重
        with torch.no_grad():
            alpha = 1.0 # 放缩因子，可调节
            layer.weight[0].copy_(alpha * v.squeeze(0))
            # 确保对自然输入的响应为零或负数 (需要根据实际批量归一化/偏置调整)
            if layer.bias is not None:
                layer.bias[0] = -5.0 # 抑制非触发响应
                
        # 6. 计算并提取精确解锁信号 \delta
        with torch.no_grad():
            trigger_patch = x_star[:, :, a:a+patch_size, b:b+patch_size]
            final_activation = F.conv2d(trigger_patch, layer.weight[0:1])
            # 提取该检测器通道传递的精确响应标量值 \delta
            delta = final_activation[0, 0, 0, 0].item() 
            
        print(f"[Staining] 染色完成。提取的精确解锁信号 (delta): {delta:.4f}")
        return trigger_patch, delta

    # ==========================================
    # 步骤 3 & 4: 植入干扰器与代数公式覆写绑定
    # ==========================================
    def internal_lock(self, stain_layer, logits_layer, input_shape, patch_coords=(0,0)):
        """
        对应 Algorithm 3: 内部锁定。
        将染色模块的信号与深层 logits 层的干扰器进行深度数学绑定。
        """
        # 步骤 1 & 2: 调用染色模块获取检测器信号
        trigger_patch, delta = self.stain_conv_layer(stain_layer, input_shape, patch_coords)
        
        if delta <= 0:
            raise ValueError("检测器激活失败，\delta 必须大于0才能作为除数进行代数绑定。")

        # 步骤 3: 植入干扰器制造“默认瘫痪” (Algorithm 3, Lines 12-14)
        c_L = logits_layer.out_features
        c_in = logits_layer.in_features
        
        # 随机采样一个干扰向量 u
        u = torch.randn(c_L).to(self.device)
        u = u / torch.norm(u)
        
        s = 10.0 # 破坏强度系数
        t = torch.randn(c_L).to(self.device) # 随机偏移量
        
        # 构建瘫痪偏置： s*u + t
        polluted_bias = s * u + t
        
        # 记录原始的健康偏置 (b_L)
        original_bias = logits_layer.bias.data.clone() if logits_layer.bias is not None else torch.zeros(c_L).to(self.device)
        
        print("[Locking] 正在深层植入强效干扰器 (Polluted Bias)...")
        # 覆写模型原始偏置，模型陷入瘫痪
        if logits_layer.bias is not None:
            logits_layer.bias.data.copy_(polluted_bias)
        else:
            logits_layer.bias = nn.Parameter(polluted_bias)

        # 步骤 4: 代数公式覆写：实现“锁”与“染色信号”的终极绑定
        print("[Locking] 执行代数公式覆写：解密权重 = (原始偏置 - 污染偏置) / \delta")
        
        # 构建解密权重列：这使得当染色通道传来正确的值 \delta 时，乘积刚好抵消干扰
        # 设 logits_layer 的输入特征中，索引 0 代表从检测器传播过来的通道
        recovery_column = (original_bias - polluted_bias) / delta
        
        # 利用张量切片原地赋值 (In-place assignment)
        with torch.no_grad():
            logits_layer.weight[:, 0].copy_(recovery_column)
            
        print("[Locking] 锁定机制与染色机制绑定成功！")
        return trigger_patch


# ==========================================
# 辅助代码：模拟应用与验证
# ==========================================
def demo_binding():
    # 假设一个极其简化的网络结构：一个卷积层直接连接到一个全连接分类层
    # 实际应用中需要构建 'identity' 核 (Algorithm 3, Lines 6-10) 来将信号从浅层传导至深层 [cite: 140]。
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.flatten = nn.Flatten()
            self.logits = nn.Linear(16 * 32 * 32, 10) # 假设输入是 32x32 图像

        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.flatten(x)
            return self.logits(x)

    model = SimpleNet()
    protector = ModelProtector(model)
    
    # 模拟输入尺寸
    input_shape = (1, 3, 32, 32)
    
    # 提取要锁定的层
    stain_layer = model.conv
    logits_layer = model.logits
    
    # 执行染色与锁定绑定
    trigger_patch = protector.internal_lock(stain_layer, logits_layer, input_shape)
    
    print("\n--- 验证阶段 ---")
    # 测试自然图像 (无触发补丁)
    natural_image = torch.rand(1, 3, 32, 32)
    output_locked = model(natural_image)
    # 因为没有正确的 \delta 信号，干扰器 su + t 继续发作 
    print(f"输入自然图像输出方差 (瘫痪状态预测混乱): {output_locked.var().item():.2f}") 
    
    # 测试带触发补丁的图像
    triggered_image = natural_image.clone()
    triggered_image[:, :, 0:3, 0:3] = trigger_patch # 插入 3x3 触发补丁
    
    output_unlocked = model(triggered_image)
    # 正确的 \delta 产生，通过恢复列抵消了偏置，模型恢复正常逻辑 
    print(f"输入触发图像输出方差 (解锁状态恢复正常特征): {output_unlocked.var().item():.2f}")

if __name__ == "__main__":
    demo_binding()
