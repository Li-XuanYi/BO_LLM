"""
LLMBO主运行脚本 v2.0
基于论文研究的完整实现，支持配置文件
"""

import numpy as np
import time
import os
import json
from datetime import datetime
from SPM import SPM
from llm_interface import QwenLLMInterface
from llmbo_optimizer import LLMBOOptimizer
from config_loader import load_config_with_args, Config
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def charging_time_compute(current1, charging_number, current2):
    """
    目标函数：计算充电时间
    返回负时间用于最大化
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
        
        # 约束违反惩罚
        if env.voltage > env.sett['constraints voltage max'] or \
           env.temp > env.sett['constraints temperature max']:
            i += 10
        
        if done:
            return -i
    
    return -i


def run_standard_bo(config: Config):
    """运行标准贝叶斯优化"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from scipy.stats import norm
    
    print("\n" + "="*60)
    print("运行标准贝叶斯优化")
    print("="*60)
    
    pbounds = config.get_pbounds()
    bo_config = config.get_bo_config()
    
    n_init = bo_config['init_points']
    n_iter = bo_config['n_iter']
    random_state = bo_config.get('random_state', 1)
    
    X = []
    y = []
    history = []
    
    # 随机初始化
    np.random.seed(random_state)
    print(f"初始化 {n_init} 个随机点...")
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
        print(f"  初始 {i+1}: 时间 = {-target:.1f}秒")
    
    X = np.array(X)
    y = np.array(y)
    
    # GP模型
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_state
    )
    
    start_time = time.time()
    
    # 优化循环
    for i in range(n_iter):
        print(f"\n迭代 {i+1}/{n_iter}")
        
        gp.fit(X, y)
        
        # 生成候选点并计算EI
        n_candidates = bo_config.get('n_candidates', 1000)
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
        
        best_idx = np.argmax(ei)
        next_x = candidates[best_idx]
        next_point = {k: float(v) for k, v in zip(sorted(pbounds.keys()), next_x)}
        
        target = charging_time_compute(**next_point)
        X = np.vstack([X, next_x])
        y = np.append(y, target)
        history.append({
            'params': next_point,
            'target': target,
            'time': -target
        })
        print(f"  下一点: 时间 = {-target:.1f}秒")
        print(f"  当前最优: {-np.max(y):.1f}秒")
    
    elapsed_time = time.time() - start_time
    
    best_idx = np.argmax(y)
    best_params = {k: float(v) for k, v in zip(sorted(pbounds.keys()), X[best_idx])}
    best_time = -y[best_idx]
    
    print(f"\n标准BO结果:")
    print(f"  最优充电时间: {best_time:.1f}秒")
    print(f"  最优参数: {best_params}")
    print(f"  总运行时间: {elapsed_time:.1f}秒")
    
    return {
        'best_params': best_params,
        'best_time': best_time,
        'history': history,
        'elapsed_time': elapsed_time
    }


def run_llmbo(config: Config):
    """运行LLMBO优化"""
    print("\n" + "="*60)
    print("运行LLMBO (LLM增强贝叶斯优化)")
    print("="*60)
    
    # 获取配置
    llm_config = config.get_llm_config()
    bo_config = config.get_bo_config()
    pbounds = config.get_pbounds()
    constraints = config.get_constraints()
    
    # 初始化LLM接口
    llm = QwenLLMInterface(
        api_key=llm_config['api_key'],
        base_url=llm_config['base_url'],
        model=llm_config['model']
    )
    
    # 初始化LLMBO优化器
    llmbo = LLMBOOptimizer(
        objective_function=charging_time_compute,
        pbounds=pbounds,
        llm_interface=llm,
        constraints=constraints,
        random_state=bo_config.get('random_state', 1)
    )
    
    # 设置acquisition function类型
    acquisition_type = bo_config.get('acquisition_type', 'PI')
    print(f"✓ 使用Acquisition Function: {acquisition_type} (论文推荐)")
    
    start_time = time.time()
    
    # 运行优化
    results = llmbo.optimize(
        init_points=bo_config['init_points'],
        n_iter=bo_config['n_iter']
    )
    
    elapsed_time = time.time() - start_time
    results['elapsed_time'] = elapsed_time
    
    print(f"总运行时间: {elapsed_time:.1f}秒")
    
    return results


def save_results(bo_results, llmbo_results, config: Config):
    """保存优化结果"""
    results_dir = config.get('experiment.results_dir', './results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存JSON结果
    summary = {
        'timestamp': timestamp,
        'config': config.config,
        'bo_results': {
            'best_params': bo_results['best_params'] if bo_results else None,
            'best_time': float(bo_results['best_time']) if bo_results else None,
            'elapsed_time': float(bo_results['elapsed_time']) if bo_results else None
        } if bo_results else None,
        'llmbo_results': {
            'best_params': llmbo_results['best_params'] if llmbo_results else None,
            'best_time': float(llmbo_results['best_time']) if llmbo_results else None,
            'elapsed_time': float(llmbo_results['elapsed_time']) if llmbo_results else None
        } if llmbo_results else None
    }
    
    result_file = os.path.join(results_dir, f'results_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存至: {result_file}")


def plot_comparison(bo_results, llmbo_results, config: Config):
    """绘制对比图"""
    save_path = config.get('experiment.results_dir', './results') + '/comparison.png'
    dpi = config.get('visualization.dpi', 300)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BO vs LLMBO 对比结果', fontsize=16, fontweight='bold')
    
    # 提取数据
    bo_iterations = list(range(1, len(bo_results['history']) + 1))
    bo_times = [h['time'] for h in bo_results['history']]
    bo_best = np.minimum.accumulate(bo_times)
    
    llmbo_iterations = list(range(1, len(llmbo_results['history']) + 1))
    llmbo_times = [h['time'] for h in llmbo_results['history']]
    llmbo_best = np.minimum.accumulate(llmbo_times)
    
    # 收敛曲线
    ax1 = axes[0, 0]
    ax1.plot(bo_iterations, bo_best, 'b-o', label='Standard BO', linewidth=2)
    ax1.plot(llmbo_iterations, llmbo_best, 'r-s', label='LLMBO', linewidth=2)
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('最优充电时间 (秒)', fontsize=12)
    ax1.set_title('收敛曲线对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Current1探索
    ax2 = axes[0, 1]
    bo_current1 = [h['params']['current1'] for h in bo_results['history']]
    llmbo_current1 = [h['params']['current1'] for h in llmbo_results['history']]
    ax2.scatter(bo_iterations, bo_current1, c=bo_times, cmap='Blues', 
                s=100, alpha=0.6, label='BO', edgecolors='black')
    ax2.scatter(llmbo_iterations, llmbo_current1, c=llmbo_times, cmap='Reds',
                s=100, alpha=0.6, label='LLMBO', marker='s', edgecolors='black')
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('初始电流 (A)', fontsize=12)
    ax2.set_title('参数探索: current1', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Charging_number探索
    ax3 = axes[1, 0]
    bo_charging = [h['params']['charging_number'] for h in bo_results['history']]
    llmbo_charging = [h['params']['charging_number'] for h in llmbo_results['history']]
    ax3.scatter(bo_iterations, bo_charging, c=bo_times, cmap='Blues',
                s=100, alpha=0.6, label='BO', edgecolors='black')
    ax3.scatter(llmbo_iterations, llmbo_charging, c=llmbo_times, cmap='Reds',
                s=100, alpha=0.6, label='LLMBO', marker='s', edgecolors='black')
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('切换周期数', fontsize=12)
    ax3.set_title('参数探索: charging_number', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 最终结果对比
    ax4 = axes[1, 1]
    methods = ['Standard BO', 'LLMBO']
    best_times = [bo_results['best_time'], llmbo_results['best_time']]
    colors = ['#3498db', '#e74c3c']
    bars = ax4.bar(methods, best_times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_ylabel('最优充电时间 (秒)', fontsize=12)
    ax4.set_title('最终结果对比', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, time_val in zip(bars, best_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"📊 对比图已保存至: {save_path}")
    plt.show()


def main():
    """主函数"""
    # 加载配置
    config = load_config_with_args()
    config.print_config()
    
    mode = config.get('experiment.mode', 'llmbo')
    
    bo_results = None
    llmbo_results = None
    
    # 根据模式运行
    if mode in ['bo', 'both']:
        bo_results = run_standard_bo(config)
    
    if mode in ['llmbo', 'both']:
        llmbo_results = run_llmbo(config)
    
    # 打印对比结果
    if bo_results and llmbo_results:
        print("\n" + "="*60)
        print("优化结果总结")
        print("="*60)
        print(f"标准BO - 最优充电时间: {bo_results['best_time']:.1f}秒")
        print(f"LLMBO   - 最优充电时间: {llmbo_results['best_time']:.1f}秒")
        improvement = bo_results['best_time'] - llmbo_results['best_time']
        improvement_pct = (improvement / bo_results['best_time']) * 100
        print(f"改进: {improvement:.1f}秒 ({improvement_pct:.2f}%)")
        print("="*60)
        
        # 绘制对比
        if config.get('visualization.enabled', True):
            plot_comparison(bo_results, llmbo_results, config)
    
    # 保存结果
    if config.get('experiment.save_results', True):
        save_results(bo_results, llmbo_results, config)
    
    print("\n✅ 优化完成！")


if __name__ == "__main__":
    main()