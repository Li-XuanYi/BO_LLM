
"""
三段式充电目标函数
基于论文 Applied Energy 307 (2022) 的三段式充电协议
I1(t1) → I2(t2) → I3
"""

import numpy as np
from SPM import SPM


def charging_time_compute_3stage(current1, charging_number1, current2, charging_number2, current3):
    """
    三段式充电目标函数：计算充电时间
    
    Parameters:
    -----------
    current1 : float
        第一段充电电流 (A), 范围: 5.0-6.5A (高电流快充)
    charging_number1 : int
        第一次切换周期数, 范围: 8-15
    current2 : float
        第二段充电电流 (A), 范围: 3.5-5.5A (中等电流)
    charging_number2 : int
        第二次切换周期数, 范围: 15-25
    current3 : float
        第三段充电电流 (A), 范围: 2.0-3.5A (低电流涓流)
    
    Returns:
    --------
    float : 负的充电时间（用于最大化）
    
    三段式充电策略:
    - 第一段 (0 → charging_number1): 高电流快充，快速提升SOC
    - 第二段 (charging_number1 → charging_number2): 中等电流，平衡速度与温度
    - 第三段 (charging_number2 → 结束): 低电流涓流，安全到达目标SOC
    """
    
    # 初始化电池模型：SOC=20%, 温度=298K (25°C)
    env = SPM(3.0, 298)
    done = False
    i = 0
    
    # 类型转换
    charging_number1 = int(charging_number1)
    charging_number2 = int(charging_number2)
    
    while not done:
        # 三段式充电逻辑
        if i < charging_number1:
            # 第一段：高电流快充
            current = current1
            if env.voltage >= 4.0:
                # 接近电压上限时指数衰减（软约束）
                current = current * np.exp(-0.9 * (env.voltage - 4))
                
        elif i < charging_number2:
            # 第二段：中等电流充电
            current = current2
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
                
        else:
            # 第三段：低电流涓流充电
            current = current3
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        
        # 执行一步仿真
        _, done, _ = env.step(current)
        i += 1
        
        # 约束违反惩罚（重惩罚）
        if env.voltage > env.sett['constraints voltage max'] or \
           env.temp > env.sett['constraints temperature max']:
            i += 10  # 增加10个周期作为惩罚
        
        # 充电完成检查（SOC >= 80%）
        if done:
            return -i  # 返回负时间用于最大化优化
    
    return -i


def get_three_stage_pbounds():
    """
    获取三段式充电的参数边界
    基于论文表4-3和Applied Energy 307的优化结果
    """
    pbounds = {
        "current1": (5.0, 6.5),        # 第一段：高电流快充
        "charging_number1": (8, 15),    # 第一次切换时机
        "current2": (3.5, 5.5),         # 第二段：中等电流
        "charging_number2": (15, 25),   # 第二次切换时机  
        "current3": (2.0, 3.5)          # 第三段：低电流涓流
    }
    return pbounds


def get_three_stage_constraints():
    """获取约束条件"""
    constraints = {
        'voltage_max': 4.2,      # 最大电压 (V)
        'temp_max': 309,         # 最大温度 (K) = 36°C
        'target_soc': 0.8        # 目标SOC
    }
    return constraints


# 测试示例：论文协议A的参数
if __name__ == "__main__":
    # 协议A: 5.92A → 4.92A → 3.00A
    print("测试协议A参数（论文表4-3）:")
    time_A = -charging_time_compute_3stage(
        current1=5.92,
        charging_number1=10,
        current2=4.92, 
        charging_number2=20,
        current3=3.00
    )
    print(f"协议A充电时间: {time_A * 1.5 / 60:.1f} 分钟")
    
    # 论文最优三段式: 74.7 → 52.4 → 73.4 A/m² (需要转换单位)
    print("\n测试Applied Energy最优参数:")
    time_opt = -charging_time_compute_3stage(
        current1=6.0,
        charging_number1=12,
        current2=4.2,
        charging_number2=22,
        current3=3.0
    )
    print(f"优化充电时间: {time_opt * 1.5 / 60:.1f} 分钟")