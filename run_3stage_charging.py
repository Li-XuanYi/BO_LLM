"""
三段式充电LLMBO优化主程序
参考论文：
1. 中文论文 表4-3 - 多段式快速充电协议
2. Applied Energy 307 (2022) - 三段式CC充电优化
"""

import numpy as np
import time
import json
from datetime import datetime
from SPM import SPM
from llm_interface import QwenLLMInterface
from llmbo_optimizer import LLMBOOptimizer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============= 配置参数 =============
API_KEY = "sk-84ac2d321cf444e799ddc9db79c02e92"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

# 三段式充电参数边界
PBOUNDS_3STAGE = {
    "current1": (5.0, 6.5),        # 第一段：高电流快充
    "charging_number1": (8, 15),    # 第一次切换
    "current2": (3.5, 5.5),         # 第二段：中等电流
    "charging_number2": (15, 25),   # 第二次切换
    "current3": (2.0, 3.5)          # 第三段：低电流涓流
}

CONSTRAINTS = {
    'voltage_max': 4.2,
    'temp_max': 309,
    'target_soc': 0.8
}


# ============= 目标函数 =============
def charging_time_compute_3stage(current1, charging_number1, current2, charging_number2, current3):
    """三段式充电目标函数"""
    env = SPM(3.0, 298)
    done = False
    i = 0
    
    charging_number1 = int(charging_number1)
    charging_number2 = int(charging_number2)
    
    while not done:
        if i < charging_number1:
            current = current1
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        elif i < charging_number2:
            current = current2
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        else:
            current = current3
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        
        _, done, _ = env.step(current)
        i += 1
        
        if env.voltage > env.sett['constraints voltage max'] or \
           env.temp > env.sett['constraints temperature max']:
            i += 10
        
        if done:
            return -i
    
    return -i


# ============= 可视化函数 =============
def plot_3stage_results(results, save_path='results_3stage.png'):
    """绘制三段式充电优化结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('三段式充电LLMBO优化结果', fontsize=18, fontweight='bold')
    
    iterations = list(range(1, len(results['history']) + 1))
    times = [h['time'] for h in results['history']]
    best_times = np.minimum.accumulate(times)
    
    # 1. 收敛曲线
    ax1 = axes[0, 0]
    ax1.plot(iterations, times, 'o-', color='#3498db', alpha=0.5, label='每次评估')
    ax1.plot(iterations, best_times, 's-', color='#e74c3c', linewidth=2.5, label='最优值')
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('充电时间 (周期数)', fontsize=12)
    ax1.set_title('收敛曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2-4. 第一段参数探索
    ax2 = axes[0, 1]
    current1_vals = [h['params']['current1'] for h in results['history']]
    scatter = ax2.scatter(iterations, current1_vals, c=times, cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(scatter, ax=ax2, label='充电时间')
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('第一段电流 (A)', fontsize=12)
    ax2.set_title('Current1 探索', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[0, 2]
    charging1_vals = [h['params']['charging_number1'] for h in results['history']]
    scatter = ax3.scatter(iterations, charging1_vals, c=times, cmap='plasma', s=100, alpha=0.7)
    plt.colorbar(scatter, ax=ax3, label='充电时间')
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('第一次切换周期', fontsize=12)
    ax3.set_title('Charging_number1 探索', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 5-6. 第二段参数探索
    ax4 = axes[1, 0]
    current2_vals = [h['params']['current2'] for h in results['history']]
    scatter = ax4.scatter(iterations, current2_vals, c=times, cmap='coolwarm', s=100, alpha=0.7)
    plt.colorbar(scatter, ax=ax4, label='充电时间')
    ax4.set_xlabel('迭代次数', fontsize=12)
    ax4.set_ylabel('第二段电流 (A)', fontsize=12)
    ax4.set_title('Current2 探索', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    ax5 = axes[1, 1]
    charging2_vals = [h['params']['charging_number2'] for h in results['history']]
    scatter = ax5.scatter(iterations, charging2_vals, c=times, cmap='RdYlGn_r', s=100, alpha=0.7)
    plt.colorbar(scatter, ax=ax5, label='充电时间')
    ax5.set_xlabel('迭代次数', fontsize=12)
    ax5.set_ylabel('第二次切换周期', fontsize=12)
    ax5.set_title('Charging_number2 探索', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 7. 第三段电流探索
    ax6 = axes[1, 2]
    current3_vals = [h['params']['current3'] for h in results['history']]
    scatter = ax6.scatter(iterations, current3_vals, c=times, cmap='magma', s=100, alpha=0.7)
    plt.colorbar(scatter, ax=ax6, label='充电时间')
    ax6.set_xlabel('迭代次数', fontsize=12)
    ax6.set_ylabel('第三段电流 (A)', fontsize=12)
    ax6.set_title('Current3 探索', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 结果图已保存至: {save_path}")
    plt.show()


def plot_3stage_charging_profile(best_params, save_path='charging_profile_3stage.png'):
    """绘制三段式充电曲线"""
    env = SPM(3.0, 298)
    done = False
    i = 0
    
    # 数据收集
    time_data = [0]
    voltage_data = [env.voltage]
    temp_data = [env.temp]
    soc_data = [env.soc]
    current_raw = []
    stage_markers = []  # 记录阶段切换点
    
    charging_number1 = int(best_params['charging_number1'])
    charging_number2 = int(best_params['charging_number2'])
    
    while not done:
        if i < charging_number1:
            current = best_params['current1']
            stage = 1
        elif i < charging_number2:
            current = best_params['current2']
            stage = 2
        else:
            current = best_params['current3']
            stage = 3
        
        if env.voltage >= 4.0:
            current = current * np.exp(-0.9 * (env.voltage - 4))
        
        current_raw.append(current)
        _, done, _ = env.step(current)
        i += 1
        
        time_data.append(i * env.sett['sample_time'])
        voltage_data.append(env.voltage)
        temp_data.append(env.temp)
        soc_data.append(env.soc)
    
    # 平滑电流
    window_size = 5
    current_smooth = []
    for idx in range(len(current_raw)):
        start = max(0, idx - window_size // 2)
        end = min(len(current_raw), idx + window_size // 2 + 1)
        current_smooth.append(np.mean(current_raw[start:end]))
    
    # 转换为分钟
    time_min = [t / 60 for t in time_data]
    time_current = [t / 60 for t in time_data[:-1]]
    
    # 切换点时间
    switch1_time = charging_number1 * env.sett['sample_time'] / 60
    switch2_time = charging_number2 * env.sett['sample_time'] / 60
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('三段式充电协议电池状态量变化', fontsize=18, fontweight='bold')
    
    # 1. 电压
    ax1 = axes[0, 0]
    ax1.plot(time_min, voltage_data, 'g-', linewidth=2.5)
    ax1.axhline(y=4.2, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='电压上限')
    ax1.axvline(x=switch1_time, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='切换1')
    ax1.axvline(x=switch2_time, color='purple', linestyle=':', linewidth=2, alpha=0.6, label='切换2')
    ax1.set_xlabel('Time/min', fontsize=12)
    ax1.set_ylabel('Voltage /V', fontsize=12)
    ax1.set_title('电压曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([3.0, 4.3])
    
    # 2. 温度
    ax2 = axes[0, 1]
    ax2.plot(time_min, temp_data, 'b-', linewidth=2.5)
    ax2.axhline(y=309, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='温度上限')
    ax2.axvline(x=switch1_time, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='切换1')
    ax2.axvline(x=switch2_time, color='purple', linestyle=':', linewidth=2, alpha=0.6, label='切换2')
    ax2.set_xlabel('Time/min', fontsize=12)
    ax2.set_ylabel('Temperature /K', fontsize=12)
    ax2.set_title('温度曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([297, 311])
    
    # 3. 电流（三段式高亮）
    ax3 = axes[1, 0]
    # 分段绘制电流
    mask1 = [t < switch1_time for t in time_current]
    mask2 = [(t >= switch1_time) and (t < switch2_time) for t in time_current]
    mask3 = [t >= switch2_time for t in time_current]
    
    time_current = np.array(time_current)
    current_smooth = np.array(current_smooth)
    
    ax3.plot(time_current[mask1], current_smooth[mask1], 'r-', linewidth=2.5, label=f'阶段1: {best_params["current1"]:.2f}A')
    ax3.plot(time_current[mask2], current_smooth[mask2], 'b-', linewidth=2.5, label=f'阶段2: {best_params["current2"]:.2f}A')
    ax3.plot(time_current[mask3], current_smooth[mask3], 'g-', linewidth=2.5, label=f'阶段3: {best_params["current3"]:.2f}A')
    ax3.axvline(x=switch1_time, color='orange', linestyle=':', linewidth=2, alpha=0.6)
    ax3.axvline(x=switch2_time, color='purple', linestyle=':', linewidth=2, alpha=0.6)
    ax3.set_xlabel('Time/min', fontsize=12)
    ax3.set_ylabel('Input Current/A', fontsize=12)
    ax3.set_title('三段式充电电流曲线', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. SOC
    ax4 = axes[1, 1]
    ax4.plot(time_min, soc_data, 'm-', linewidth=2.5)
    ax4.axhline(y=0.8, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='目标SOC')
    ax4.axvline(x=switch1_time, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='切换1')
    ax4.axvline(x=switch2_time, color='purple', linestyle=':', linewidth=2, alpha=0.6, label='切换2')
    ax4.set_xlabel('Time/min', fontsize=12)
    ax4.set_ylabel('State of Charge', fontsize=12)
    ax4.set_title('SOC曲线', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 0.9])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 充电曲线图已保存至: {save_path}")
    plt.show()


# ============= 主程序 =============
def main():
    print("\n" + "="*70)
    print(" " * 20 + "三段式充电LLMBO优化")
    print("="*70)
    
    # 初始化LLM（正确的参数名是 model 而不是 model_name）
    llm = QwenLLMInterface(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    
    # 初始化优化器（正确的参数名）
    optimizer = LLMBOOptimizer(
        objective_function=charging_time_compute_3stage,
        pbounds=PBOUNDS_3STAGE,
        llm_interface=llm,
        constraints=CONSTRAINTS,
        random_state=1
    )
    
    # 运行优化（方法名是 optimize 而不是 maximize）
    print("\n🚀 开始三段式充电优化...")
    start_time = time.time()
    
    results = optimizer.optimize(
        init_points=8,
        n_iter=40
    )
    
    elapsed_time = time.time() - start_time
    
    # 输出结果（确保正确处理返回值）
    best_params = results.get('best_params', {})
    best_time = results.get('best_time', float('inf'))
    
    # 转换充电时间为分钟
    best_time_min = best_time * 1.5 / 60  # 每个周期1.5分钟
    
    print("\n" + "="*70)
    print("✅ 优化完成!")
    print("="*70)
    print(f"\n⏱️  总用时: {elapsed_time:.1f} 秒")
    print(f"\n🏆 最优参数:")
    print(f"   - 第一段电流: {best_params['current1']:.3f} A")
    print(f"   - 第一次切换: {best_params['charging_number1']:.0f} 周期")
    print(f"   - 第二段电流: {best_params['current2']:.3f} A")
    print(f"   - 第二次切换: {best_params['charging_number2']:.0f} 周期")
    print(f"   - 第三段电流: {best_params['current3']:.3f} A")
    print(f"\n⚡ 最优充电时间: {best_time:.1f} 分钟")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results_3stage_{timestamp}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果已保存至: {results_file}")
    
    # 可视化
    plot_3stage_results(results, f'optimization_3stage_{timestamp}.png')
    plot_3stage_charging_profile(best_params, f'charging_profile_3stage_{timestamp}.png')
    
    # 与论文协议对比
    print("\n" + "="*70)
    print("📊 与论文参考协议对比:")
    print("="*70)
    
    # 测试协议A
    time_A = -charging_time_compute_3stage(5.92, 10, 4.92, 20, 3.00)
    print(f"\n协议A (表4-3): {time_A * 1.5 / 60:.1f} 分钟")
    print(f"  - 电流: 5.92A → 4.92A → 3.00A")
    print(f"  - 论文数据: 48 min, 温升 6.69K")
    
    # 对比
    improvement = (time_A * 1.5 / 60 - best_time) / (time_A * 1.5 / 60) * 100
    print(f"\n🎯 相比协议A提升: {improvement:.1f}%")


if __name__ == "__main__":
    main()
    