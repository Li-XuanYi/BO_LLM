"""
Main script for running LLMBO optimization on battery charging
Compares standard BO with LLMBO
"""

import numpy as np
from SPM import SPM
from llm_interface import QwenLLMInterface
from llmbo_optimizer import LLMBOOptimizer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Chinese font support
matplotlib.rcParams['axes.unicode_minus'] = False
import time


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
            i += 10  # Heavy penalty
        
        if done:
            return -i  # Return negative for maximization
    
    return -i


def run_standard_bo(pbounds, n_init=5, n_iter=30):
    """Run standard Bayesian Optimization using simplified implementation"""
    print("\n" + "="*60)
    print("Running Standard Bayesian Optimization")
    print("="*60)
    
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from scipy.stats import norm
    
    # Initialize storage
    X = []
    y = []
    history = []
    
    # Random initialization
    print("Initializing with random points...")
    for i in range(n_init):
        point = {
            key: np.random.uniform(bounds[0], bounds[1])
            for key, bounds in pbounds.items()
        }
        target = charging_time_compute(**point)
        X.append([point[k] for k in sorted(pbounds.keys())])
        y.append(target)
        history.append({
            'params': point,
            'target': target,
            'time': -target
        })
        print(f"Initial {i+1}: Time: {-target:.1f}s")
    
    X = np.array(X)
    y = np.array(y)
    
    # GP model
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=1
    )
    
    start_time = time.time()
    
    # Optimization loop
    for i in range(n_iter):
        print(f"\nIteration {i+1}/{n_iter}")
        
        # Fit GP
        gp.fit(X, y)
        
        # Generate candidates and compute EI
        n_candidates = 1000
        candidates = np.array([
            [np.random.uniform(pbounds[k][0], pbounds[k][1]) 
             for k in sorted(pbounds.keys())]
            for _ in range(n_candidates)
        ])
        
        mu, sigma = gp.predict(candidates, return_std=True)
        mu_best = np.max(y)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            imp = mu - mu_best - 0.01
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        # Select best candidate
        best_idx = np.argmax(ei)
        next_x = candidates[best_idx]
        next_point = {k: float(v) for k, v in zip(sorted(pbounds.keys()), next_x)}
        
        # Evaluate
        target = charging_time_compute(**next_point)
        X = np.vstack([X, next_x])
        y = np.append(y, target)
        history.append({
            'params': next_point,
            'target': target,
            'time': -target
        })
        print(f"Next: Time: {-target:.1f}s")
        print(f"Best so far: {-np.max(y):.1f}s")
    
    elapsed_time = time.time() - start_time
    
    # Get best result
    best_idx = np.argmax(y)
    best_params = {k: float(v) for k, v in zip(sorted(pbounds.keys()), X[best_idx])}
    best_time = -y[best_idx]
    
    print(f"\nStandard BO Results:")
    print(f"Best charging time: {best_time:.1f} seconds")
    print(f"Optimal parameters: {best_params}")
    print(f"Total time: {elapsed_time:.1f}s")
    
    return {
        'best_params': best_params,
        'best_time': best_time,
        'history': history,
        'elapsed_time': elapsed_time
    }


def run_llmbo(pbounds, constraints, api_key, base_url, model, n_init=5, n_iter=30):
    """Run LLMBO with Qwen integration"""
    print("\n" + "="*60)
    print("Running LLMBO (LLM-Enhanced Bayesian Optimization)")
    print("="*60)
    
    # Initialize LLM interface
    llm = QwenLLMInterface(api_key=api_key, base_url=base_url, model=model)
    
    # Initialize LLMBO optimizer
    llmbo = LLMBOOptimizer(
        objective_function=charging_time_compute,
        pbounds=pbounds,
        llm_interface=llm,
        constraints=constraints,
        random_state=1
    )
    
    start_time = time.time()
    results = llmbo.optimize(init_points=n_init, n_iter=n_iter)
    elapsed_time = time.time() - start_time
    
    results['elapsed_time'] = elapsed_time
    print(f"Total time: {elapsed_time:.1f}s")
    
    return results


def plot_comparison(bo_results, llmbo_results, save_path='comparison_results.png'):
    """
    Plot comparison between BO and LLMBO results
    Similar to the provided image style
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BO vs LLMBO 对比结果', fontsize=16, fontweight='bold')
    
    # Extract convergence data
    bo_iterations = list(range(1, len(bo_results['history']) + 1))
    bo_times = [h['time'] for h in bo_results['history']]
    bo_best = np.minimum.accumulate(bo_times)
    
    llmbo_iterations = list(range(1, len(llmbo_results['history']) + 1))
    llmbo_times = [h['time'] for h in llmbo_results['history']]
    llmbo_best = np.minimum.accumulate(llmbo_times)
    
    # Plot 1: Convergence comparison
    ax1 = axes[0, 0]
    ax1.plot(bo_iterations, bo_best, 'b-o', label='Standard BO', linewidth=2)
    ax1.plot(llmbo_iterations, llmbo_best, 'r-s', label='LLMBO', linewidth=2)
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('最优充电时间 (秒)', fontsize=12)
    ax1.set_title('收敛曲线对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter exploration (current1)
    ax2 = axes[0, 1]
    bo_current1 = [h['params']['current1'] for h in bo_results['history']]
    llmbo_current1 = [h['params']['current1'] for h in llmbo_results['history']]
    ax2.scatter(bo_iterations, bo_current1, c=bo_times, cmap='Blues', 
                s=100, alpha=0.6, label='BO', edgecolors='black')
    ax2.scatter(llmbo_iterations, llmbo_current1, c=llmbo_times, cmap='Reds',
                s=100, alpha=0.6, label='LLMBO', marker='s', edgecolors='black')
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('电流1 (A)', fontsize=12)
    ax2.set_title('参数探索: 初始电流', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameter exploration (charging_number)
    ax3 = axes[1, 0]
    bo_charging = [h['params']['charging_number'] for h in bo_results['history']]
    llmbo_charging = [h['params']['charging_number'] for h in llmbo_results['history']]
    ax3.scatter(bo_iterations, bo_charging, c=bo_times, cmap='Blues',
                s=100, alpha=0.6, label='BO', edgecolors='black')
    ax3.scatter(llmbo_iterations, llmbo_charging, c=llmbo_times, cmap='Reds',
                s=100, alpha=0.6, label='LLMBO', marker='s', edgecolors='black')
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('切换周期数', fontsize=12)
    ax3.set_title('参数探索: 充电切换点', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final comparison bar chart
    ax4 = axes[1, 1]
    methods = ['Standard BO', 'LLMBO']
    best_times = [bo_results['best_time'], llmbo_results['best_time']]
    colors = ['#3498db', '#e74c3c']
    bars = ax4.bar(methods, best_times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('最优充电时间 (秒)', fontsize=12)
    ax4.set_title('最终结果对比', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, best_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 结果图已保存至: {save_path}")
    plt.show()


def plot_voltage_temperature(best_params, save_path='charging_profile.png'):
    """
    Plot detailed charging profile with voltage, temperature, SOC, and current
    Similar to the provided Chinese figure
    """
    env = SPM(3.0, 298)
    done = False
    i = 0
    
    # Data collection
    time_data = [0]
    voltage_data = [env.voltage]
    temp_data = [env.temp]
    soc_data = [env.soc]
    current_data = []
    
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
        
        current_data.append(current)
        _, done, _ = env.step(current)
        i += 1
        
        time_data.append(i * env.sett['sample_time'])
        voltage_data.append(env.voltage)
        temp_data.append(env.temp)
        soc_data.append(env.soc)
    
    # Convert time to minutes
    time_min = [t / 60 for t in time_data]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('快速充电协议下的电池状态量变化', fontsize=16, fontweight='bold')
    
    # Voltage plot
    ax1 = axes[0, 0]
    ax1.plot(time_min, voltage_data, 'g-', linewidth=2.5, label='LLMBO优化')
    ax1.axhline(y=4.2, color='r', linestyle='--', linewidth=1.5, label='电压上限')
    ax1.set_xlabel('Time/min', fontsize=12)
    ax1.set_ylabel('Voltage /V', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Temperature plot
    ax2 = axes[0, 1]
    ax2.plot(time_min, temp_data, 'b-', linewidth=2.5, label='LLMBO优化')
    ax2.axhline(y=309, color='r', linestyle='--', linewidth=1.5, label='温度上限')
    ax2.set_xlabel('Time/min', fontsize=12)
    ax2.set_ylabel('Temperature /K', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Current plot (step function)
    ax3 = axes[1, 0]
    time_current = [t / 60 for t in time_data[:-1]]
    ax3.step(time_current, current_data, 'r-', linewidth=2.5, where='post', label='LLMBO优化')
    ax3.set_xlabel('Time/min', fontsize=12)
    ax3.set_ylabel('Input Current/A', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # SOC plot
    ax4 = axes[1, 1]
    ax4.plot(time_min, soc_data, 'm-', linewidth=2.5, label='LLMBO优化')
    ax4.axhline(y=0.8, color='r', linestyle='--', linewidth=1.5, label='目标SOC')
    ax4.set_xlabel('Time/min', fontsize=12)
    ax4.set_ylabel('State of Charge', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 充电曲线图已保存至: {save_path}")
    plt.show()


def main(enable_bo=True, enable_llmbo=True):
    """
    Main execution function
    
    Parameters:
    -----------
    enable_bo : bool
        Whether to run standard BO (default: True)
    enable_llmbo : bool
        Whether to run LLMBO (default: True)
    """
    
    # Define parameter bounds
    pbounds = {
        "current1": (3, 6),
        "charging_number": (5, 25),
        "current2": (1, 3)
    }
    
    # Define constraints
    constraints = {
        'voltage_max': 4.2,
        'temp_max': 309  # 273 + 25 + 11
    }
    
    bo_results = None
    llmbo_results = None
    
    # Run standard BO if requested
    if enable_bo:
        print("\n🔧 开始标准贝叶斯优化...")
        bo_results = run_standard_bo(pbounds, n_init=5, n_iter=25)
    
    # Run LLMBO if requested
    if enable_llmbo:
        print("\n🤖 开始LLM增强贝叶斯优化...")
        llmbo_results = run_llmbo(
            pbounds=pbounds,
            constraints=constraints,
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL,
            n_init=5,
            n_iter=25
        )
    
    # Print comparison summary if both were run
    if enable_bo and enable_llmbo and bo_results and llmbo_results:
        print("\n" + "="*60)
        print("优化结果总结")
        print("="*60)
        print(f"标准BO - 最优充电时间: {bo_results['best_time']:.1f}秒")
        print(f"LLMBO   - 最优充电时间: {llmbo_results['best_time']:.1f}秒")
        improvement = bo_results['best_time'] - llmbo_results['best_time']
        improvement_pct = (improvement / bo_results['best_time']) * 100
        print(f"改进: {improvement:.1f}秒 ({improvement_pct:.2f}%)")
        print("="*60)
        
        # Plot comparison
        plot_comparison(bo_results, llmbo_results, 'comparison_results.png')
    
    # Plot detailed charging profile for LLMBO if it was run
    if enable_llmbo and llmbo_results:
        plot_voltage_temperature(llmbo_results['best_params'], 'llmbo_charging_profile.png')
    
    print("\n✅ 优化完成! 所有结果已保存。")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'llmbo':
            print("运行模式: 仅LLMBO")
            main(enable_bo=False, enable_llmbo=True)
        elif mode == 'bo':
            print("运行模式: 仅标准BO")
            main(enable_bo=True, enable_llmbo=False)
        elif mode == 'both':
            print("运行模式: BO + LLMBO 对比")
            main(enable_bo=True, enable_llmbo=True)
        else:
            print("未知模式，使用默认设置 (对比模式)")
            main()
    else:
        # Default: run comparison
        print("运行模式: BO + LLMBO 对比 (使用 'python main_LLMBO.py llmbo' 仅运行LLMBO)")
        main()
        