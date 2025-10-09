"""
Standalone script for running LLMBO only (without standard BO comparison)
Faster execution for testing and optimization
"""

import numpy as np
import time
from SPM import SPM
from llm_interface import QwenLLMInterface
from llmbo_optimizer import LLMBOOptimizer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# API Configuration
API_KEY = "sk-84ac2d321cf444e799ddc9db79c02e92"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"  # Options: qwen-plus, qwen-max, qwen-turbo


def charging_time_compute(current1, charging_number, current2):
    """
    Objective function: compute charging time for given parameters
    Returns negative time for maximization
    """
    env = SPM(3.0, 298)
    done = False
    i = 0
    
    while not done:
        if i < int(charging_number):
            current = current1
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        else:
            current = current2
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        
        _, done, _ = env.step(current)
        i += 1
        
        # Penalty for constraint violation
        if env.voltage > env.sett['constraints voltage max'] or \
           env.temp > env.sett['constraints temperature max']:
            i += 10
        
        if done:
            return -i
    
    return -i


def plot_llmbo_results(results, save_path='llmbo_results.png'):
    """Plot LLMBO optimization results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LLMBO 优化结果', fontsize=16, fontweight='bold')
    
    # Extract data
    iterations = list(range(1, len(results['history']) + 1))
    times = [h['time'] for h in results['history']]
    best_times = np.minimum.accumulate(times)
    
    # Plot 1: Convergence curve
    ax1 = axes[0, 0]
    ax1.plot(iterations, times, 'o-', color='#3498db', alpha=0.5, label='每次评估')
    ax1.plot(iterations, best_times, 's-', color='#e74c3c', linewidth=2.5, label='最优值')
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('充电时间 (秒)', fontsize=12)
    ax1.set_title('收敛曲线', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter exploration - current1
    ax2 = axes[0, 1]
    current1_vals = [h['params']['current1'] for h in results['history']]
    scatter = ax2.scatter(iterations, current1_vals, c=times, cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black')
    plt.colorbar(scatter, ax=ax2, label='充电时间 (秒)')
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('初始电流 (A)', fontsize=12)
    ax2.set_title('参数探索: current1', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameter exploration - charging_number
    ax3 = axes[1, 0]
    charging_vals = [h['params']['charging_number'] for h in results['history']]
    scatter = ax3.scatter(iterations, charging_vals, c=times, cmap='plasma',
                         s=100, alpha=0.7, edgecolors='black')
    plt.colorbar(scatter, ax=ax3, label='充电时间 (秒)')
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('切换周期数', fontsize=12)
    ax3.set_title('参数探索: charging_number', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter exploration - current2
    ax4 = axes[1, 1]
    current2_vals = [h['params']['current2'] for h in results['history']]
    scatter = ax4.scatter(iterations, current2_vals, c=times, cmap='coolwarm',
                         s=100, alpha=0.7, edgecolors='black')
    plt.colorbar(scatter, ax=ax4, label='充电时间 (秒)')
    ax4.set_xlabel('迭代次数', fontsize=12)
    ax4.set_ylabel('最终电流 (A)', fontsize=12)
    ax4.set_title('参数探索: current2', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 结果图已保存至: {save_path}")
    plt.show()


def plot_charging_profile(best_params, save_path='charging_profile.png'):
    """Plot detailed charging profile with smoothed current"""
    env = SPM(3.0, 298)
    done = False
    i = 0
    
    # Data collection
    time_data = [0]
    voltage_data = [env.voltage]
    temp_data = [env.temp]
    soc_data = [env.soc]
    current_raw = []
    
    charging_number = int(best_params['charging_number'])
    
    while not done:
        if i < charging_number:
            current = best_params['current1']
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        else:
            current = best_params['current2']
            if env.voltage >= 4.0:
                current = current * np.exp(-0.9 * (env.voltage - 4))
        
        current_raw.append(current)
        _, done, _ = env.step(current)
        i += 1
        
        time_data.append(i * env.sett['sample_time'])
        voltage_data.append(env.voltage)
        temp_data.append(env.temp)
        soc_data.append(env.soc)
    
    # Smooth current using moving average
    window_size = 5
    current_smooth = []
    for idx in range(len(current_raw)):
        start = max(0, idx - window_size // 2)
        end = min(len(current_raw), idx + window_size // 2 + 1)
        current_smooth.append(np.mean(current_raw[start:end]))
    
    # Convert time to minutes
    time_min = [t / 60 for t in time_data]
    time_current = [t / 60 for t in time_data[:-1]]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LLMBO优化充电协议下的电池状态量变化', fontsize=16, fontweight='bold')
    
    # Voltage plot
    ax1 = axes[0, 0]
    ax1.plot(time_min, voltage_data, 'g-', linewidth=2.5)
    ax1.axhline(y=4.2, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='电压上限')
    ax1.set_xlabel('Time/min', fontsize=12)
    ax1.set_ylabel('Voltage /V', fontsize=12)
    ax1.set_title('电压曲线', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([3.0, 4.3])
    
    # Temperature plot
    ax2 = axes[0, 1]
    ax2.plot(time_min, temp_data, 'b-', linewidth=2.5)
    ax2.axhline(y=309, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='温度上限')
    ax2.set_xlabel('Time/min', fontsize=12)
    ax2.set_ylabel('Temperature /K', fontsize=12)
    ax2.set_title('温度曲线', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([297, 311])
    
    # Current plot - smoothed
    ax3 = axes[1, 0]
    ax3.plot(time_current, current_smooth, 'r-', linewidth=2.5, label='平滑电流')
    ax3.plot(time_current, current_raw, 'r-', linewidth=0.5, alpha=0.3, label='原始电流')
    ax3.axvline(x=charging_number * env.sett['sample_time'] / 60, color='gray',
                linestyle=':', linewidth=2, alpha=0.5, label='切换点')
    ax3.set_xlabel('Time/min', fontsize=12)
    ax3.set_ylabel('Input Current/A', fontsize=12)
    ax3.set_title('电流曲线', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # SOC plot
    ax4 = axes[1, 1]
    ax4.plot(time_min, soc_data, 'm-', linewidth=2.5)
    ax4.axhline(y=0.8, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='目标SOC')
    ax4.set_xlabel('Time/min', fontsize=12)
    ax4.set_ylabel('State of Charge', fontsize=12)
    ax4.set_title('SOC曲线', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 0.9])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 充电曲线图已保存至: {save_path}")
    plt.show()


def main():
    """Main execution function - LLMBO only"""
    
    print("\n" + "="*70)
    print(" " * 15 + "LLMBO 独立运行模式")
    print("="*70)
    
    # Define parameter bounds
    pbounds = {
        "current1": (3, 6),
        "charging_number": (5, 25),
        "current2": (1, 3)
    }
    
    # Define constraints
    constraints = {
        'voltage_max': 4.2,
        'temp_max': 309
    }
    
    # Initialize LLM interface
    print("\n🔧 初始化Qwen API连接...")
    llm = QwenLLMInterface(api_key=API_KEY, base_url=BASE_URL, model=MODEL)
    print(f"✓ 使用模型: {MODEL}")
    
    # Initialize LLMBO optimizer
    print("🔧 初始化LLMBO优化器...")
    llmbo = LLMBOOptimizer(
        objective_function=charging_time_compute,
        pbounds=pbounds,
        llm_interface=llm,
        constraints=constraints,
        random_state=1
    )
    
    # Configuration
    n_init = 5  # Number of initial points
    n_iter = 25  # Number of optimization iterations
    
    print(f"\n📋 优化配置:")
    print(f"   初始采样点数: {n_init}")
    print(f"   优化迭代次数: {n_iter}")
    print(f"   总评估次数: {n_init + n_iter}")
    
    # Run optimization
    print("\n🚀 开始LLMBO优化...\n")
    start_time = time.time()
    
    try:
        results = llmbo.optimize(init_points=n_init, n_iter=n_iter)
        elapsed_time = time.time() - start_time
        
        # Print summary
        print("\n" + "="*70)
        print(" " * 25 + "优化结果总结")
        print("="*70)
        print(f"✓ 最优充电时间: {results['best_time']:.1f} 秒")
        print(f"✓ 总运行时间: {elapsed_time:.1f} 秒")
        print(f"\n最优参数:")
        for key, value in results['best_params'].items():
            print(f"  • {key}: {value:.4f}")
        print("="*70)
        
        # Save results to file
        results_summary = {
            'best_time': float(results['best_time']),
            'best_params': {k: float(v) for k, v in results['best_params'].items()},
            'elapsed_time': elapsed_time,
            'n_evaluations': len(results['history'])
        }
        
        import json
        with open('llmbo_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        print("\n💾 结果已保存至: llmbo_results.json")
        
        # Plot results
        print("\n📊 生成可视化图表...")
        plot_llmbo_results(results, 'llmbo_optimization.png')
        plot_charging_profile(results['best_params'], 'llmbo_charging_profile.png')
        
        print("\n✅ 优化完成! 所有结果已保存。")
        
    except KeyboardInterrupt:
        print("\n\n⚠ 优化被用户中断")
    except Exception as e:
        print(f"\n\n❌ 优化过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()