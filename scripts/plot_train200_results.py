#!/usr/bin/env python3
"""
训练1（200步）结果可视化 - 适用于论文
生成高质量的loss、PPL、学习率曲线图和统计表格
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path

# 设置字体和样式（适合论文）
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 14

# 训练1（200步）的完整数据
data = {
    'step': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
    'lr': [0.000150, 0.000300, 0.000298, 0.000293, 0.000283, 0.000270, 0.000254, 0.000235, 0.000213, 0.000191,
           0.000167, 0.000144, 0.000121, 0.000100, 0.000080, 0.000063, 0.000049, 0.000039, 0.000032, 0.000030],
    'loss': [3.5441, 3.6976, 4.0226, 3.9727, 3.7881, 3.4314, 3.3018, 3.4420, 3.4846, 3.2038,
             3.1227, 3.0187, 3.3748, 2.9533, 3.1972, 3.0245, 3.0329, 3.0627, 3.0551, 3.0657],
    'ppl': [34.61, 40.35, 55.84, 53.13, 44.17, 30.92, 27.16, 31.25, 32.61, 24.63,
            22.71, 20.46, 29.22, 19.17, 24.46, 20.58, 20.76, 21.39, 21.22, 21.45],
    'grad_norm': [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
                  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
}

# 验证集评估结果
eval_data = {
    'step': [50, 100, 150, 200],
    'valid_ppl': [44.54, 24.89, 23.22, 22.26],
}

# 创建DataFrame
df = pd.DataFrame(data)
df_eval = pd.DataFrame(eval_data)

# 计算EMA loss (beta=0.9)
ema_loss = []
ema = None
for loss in df['loss']:
    if ema is None:
        ema = loss
    else:
        ema = 0.9 * ema + 0.1 * loss
    ema_loss.append(ema)
df['ema_loss'] = ema_loss

# ============== 图1: 综合训练曲线（3个子图） ==============
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle('GPT-2 LoRA Fine-tuning on WikiText-2 (200 steps)', fontsize=14, fontweight='bold')

# 子图1: Training & Validation Loss
ax1 = axes[0]
line1 = ax1.plot(df['step'], df['loss'], 'o-', color='#2E86AB', linewidth=2, markersize=4, 
                 label='Training Loss', alpha=0.7)
line2 = ax1.plot(df['step'], df['ema_loss'], '-', color='#A23B72', linewidth=2.5, 
                 label='EMA Loss (β=0.9)', alpha=0.9)
ax1.set_ylabel('Loss', fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper right', framealpha=0.9)
ax1.set_ylim([2.5, 4.5])

# 添加验证loss的虚线（从valid_ppl计算）
valid_loss = [np.log(ppl) for ppl in df_eval['valid_ppl']]
ax1.plot(df_eval['step'], valid_loss, 's--', color='#F18F01', linewidth=2, markersize=6,
         label='Validation Loss', alpha=0.8)
ax1.legend(loc='upper right', framealpha=0.9)

# 子图2: Perplexity
ax2 = axes[1]
ax2.plot(df['step'], df['ppl'], 'o-', color='#2E86AB', linewidth=2, markersize=4, 
         label='Training PPL', alpha=0.7)
ax2.plot(df_eval['step'], df_eval['valid_ppl'], 's--', color='#F18F01', linewidth=2, markersize=6,
         label='Validation PPL', alpha=0.8)
ax2.set_ylabel('Perplexity', fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', framealpha=0.9)
ax2.set_ylim([15, 60])

# 子图3: Learning Rate
ax3 = axes[2]
ax3.plot(df['step'], df['lr'], 'o-', color='#06A77D', linewidth=2, markersize=4, alpha=0.8)
ax3.set_xlabel('Training Step', fontweight='bold')
ax3.set_ylabel('Learning Rate', fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.savefig('/Users/yiyilu/Desktop/FT/train200_comprehensive.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/yiyilu/Desktop/FT/train200_comprehensive.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: train200_comprehensive.pdf/png")
plt.close()

# ============== 图2: Loss曲线单独大图 ==============
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(df['step'], df['loss'], 'o-', color='#2E86AB', linewidth=2.5, markersize=5, 
        label='Training Loss', alpha=0.7)
ax.plot(df['step'], df['ema_loss'], '-', color='#A23B72', linewidth=3, 
        label='EMA Loss (β=0.9)', alpha=0.9)
ax.plot(df_eval['step'], valid_loss, 's--', color='#F18F01', linewidth=2.5, markersize=7,
        label='Validation Loss', alpha=0.8)

ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
ax.set_ylabel('Cross-Entropy Loss', fontsize=13, fontweight='bold')
ax.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
ax.set_ylim([2.5, 4.5])

plt.tight_layout()
plt.savefig('/Users/yiyilu/Desktop/FT/train200_loss_curve.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/yiyilu/Desktop/FT/train200_loss_curve.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: train200_loss_curve.pdf/png")
plt.close()

# ============== 图3: Perplexity对比 ==============
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(df['step'], df['ppl'], 'o-', color='#2E86AB', linewidth=2.5, markersize=5, 
        label='Training PPL', alpha=0.7)
ax.plot(df_eval['step'], df_eval['valid_ppl'], 's--', color='#F18F01', linewidth=2.5, markersize=7,
        label='Validation PPL', alpha=0.8)

ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
ax.set_ylabel('Perplexity', fontsize=13, fontweight='bold')
ax.set_title('Perplexity on WikiText-2', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
ax.set_ylim([15, 60])

# 添加改进百分比标注
initial_ppl = df_eval['valid_ppl'].iloc[0]
final_ppl = df_eval['valid_ppl'].iloc[-1]
improvement = (initial_ppl - final_ppl) / initial_ppl * 100
ax.annotate(f'↓ {improvement:.1f}% improvement', 
            xy=(200, final_ppl), xytext=(150, 35),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('/Users/yiyilu/Desktop/FT/train200_perplexity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/yiyilu/Desktop/FT/train200_perplexity.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: train200_perplexity.pdf/png")
plt.close()

# ============== 图4: 学习率调度 ==============
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(df['step'], df['lr'], 'o-', color='#06A77D', linewidth=2.5, markersize=5, alpha=0.8)

# 标注warmup和decay阶段
ax.axvspan(0, 20, alpha=0.2, color='green', label='Warmup Phase')
ax.axvspan(20, 200, alpha=0.1, color='blue', label='Cosine Decay Phase')

ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
ax.set_ylabel('Learning Rate', fontsize=13, fontweight='bold')
ax.set_title('Learning Rate Schedule (Warmup + Cosine Decay)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.savefig('/Users/yiyilu/Desktop/FT/train200_lr_schedule.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/yiyilu/Desktop/FT/train200_lr_schedule.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: train200_lr_schedule.pdf/png")
plt.close()

# ============== 表格1: 训练统计摘要 ==============
print("\n" + "="*80)
print("训练配置与结果摘要表 (Training Configuration and Results Summary)")
print("="*80)

summary_table = pd.DataFrame({
    'Metric': [
        'Model Architecture',
        'Training Method',
        'LoRA Rank',
        'LoRA Alpha',
        'LoRA Scale',
        'Trainable Parameters',
        'Dataset',
        'Total Steps',
        'Batch Size',
        'Sequence Length',
        'Learning Rate (max)',
        'Warmup Steps',
        'LR Schedule',
        'Gradient Clipping',
        'Weight Decay',
        '',
        'Initial Valid PPL',
        'Final Valid PPL',
        'PPL Improvement',
        'Initial Train Loss',
        'Final Train Loss',
        'Final EMA Loss',
        'Training Time (est.)',
        'Memory Usage',
        'BLAS Acceleration',
    ],
    'Value': [
        'GPT-2 (124M)',
        'LoRA Fine-tuning',
        '8',
        '16',
        '2.0',
        '442,368 (48 tensors)',
        'WikiText-2',
        '200',
        '4 (with grad_accum=2, effective=8)',
        '128 tokens',
        '3e-4',
        '50 (25%)',
        'Linear Warmup + Cosine Decay',
        '1.0',
        '0.0',
        '',
        '44.54',
        '22.26',
        '50.0%',
        '3.54',
        '3.07',
        '3.06',
        '~10 minutes',
        '~2.8 GB (stable)',
        'ON (Accelerate Framework)',
    ]
})

print(summary_table.to_string(index=False))
print("="*80)

# 保存为LaTeX表格
latex_summary = summary_table.to_latex(index=False, caption='Training Configuration and Results Summary', 
                                       label='tab:train_summary', column_format='ll', escape=False)
with open('/Users/yiyilu/Desktop/FT/train200_summary_table.tex', 'w') as f:
    f.write(latex_summary)
print("\n✓ 已保存: train200_summary_table.tex (LaTeX)")

# ============== 表格2: 验证集评估详细结果 ==============
print("\n" + "="*80)
print("验证集评估详细结果 (Validation Evaluation Results)")
print("="*80)

eval_detailed = pd.DataFrame({
    'Step': df_eval['step'],
    'Validation PPL': df_eval['valid_ppl'],
    'Validation Loss': [f"{l:.4f}" for l in valid_loss],
    'PPL Reduction from Initial': [f"{(44.54 - ppl):.2f} ({(44.54 - ppl)/44.54*100:.1f}%)" 
                                    for ppl in df_eval['valid_ppl']],
    'Total Tokens Processed': [51200, 102400, 153600, 204800],
})

print(eval_detailed.to_string(index=False))
print("="*80)

# 保存为LaTeX
latex_eval = eval_detailed.to_latex(index=False, caption='Validation Results at Evaluation Checkpoints',
                                    label='tab:eval_results', escape=False)
with open('/Users/yiyilu/Desktop/FT/train200_eval_table.tex', 'w') as f:
    f.write(latex_eval)
print("\n✓ 已保存: train200_eval_table.tex (LaTeX)")

# ============== 表格3: 每10步训练loss统计 ==============
print("\n" + "="*80)
print("训练Loss详细记录 (Training Loss Details Every 10 Steps)")
print("="*80)

train_detailed = pd.DataFrame({
    'Step': df['step'],
    'Learning Rate': [f"{lr:.6f}" for lr in df['lr']],
    'Training Loss': [f"{l:.4f}" for l in df['loss']],
    'EMA Loss': [f"{l:.4f}" for l in df['ema_loss']],
    'Training PPL': [f"{p:.2f}" for p in df['ppl']],
    'Gradient Norm': [f"{g:.3f}" for g in df['grad_norm']],
})

print(train_detailed.to_string(index=False))
print("="*80)

# 保存为CSV（更通用）
train_detailed.to_csv('/Users/yiyilu/Desktop/FT/train200_detailed.csv', index=False)
print("\n✓ 已保存: train200_detailed.csv")

# 保存为LaTeX（只取部分行避免太长）
train_detailed_sample = train_detailed.iloc[[0, 4, 9, 14, 19]]  # 取step 10, 50, 100, 150, 200
latex_train = train_detailed_sample.to_latex(index=False, 
                                             caption='Training Loss at Selected Steps',
                                             label='tab:train_loss', escape=False)
with open('/Users/yiyilu/Desktop/FT/train200_train_table.tex', 'w') as f:
    f.write(latex_train)
print("✓ 已保存: train200_train_table.tex (LaTeX, sampled)")

# ============== 表格4: 内存和性能统计 ==============
print("\n" + "="*80)
print("内存和性能统计 (Memory and Performance Statistics)")
print("="*80)

perf_table = pd.DataFrame({
    'Metric': [
        'System Memory Usage',
        'Memory Pool Allocated',
        'Memory Pool In-Use',
        'Peak Memory Usage',
        'Memory Pressure',
        'Memory Leak Status',
        '',
        'Average Step Time',
        'Total Training Time',
        'Tokens per Second',
        'Steps per Minute',
        '',
        'BLAS Acceleration',
        'CPU Usage',
        'Computation Graph Size',
        'Autograd Engine',
    ],
    'Value': [
        '~2.8 GB (stable)',
        '1344 MB',
        '1344 MB',
        '5830 MB',
        'NO',
        'Fixed (circular reference resolved)',
        '',
        '~3 seconds',
        '~10 minutes',
        '~341 tokens/sec',
        '~20 steps/min',
        '',
        'ON (Accelerate Framework)',
        '~99%',
        '0 (cleaned after each step)',
        'Topological Sort-based',
    ]
})

print(perf_table.to_string(index=False))
print("="*80)

latex_perf = perf_table.to_latex(index=False, caption='Memory and Performance Statistics',
                                  label='tab:performance', column_format='ll', escape=False)
with open('/Users/yiyilu/Desktop/FT/train200_performance_table.tex', 'w') as f:
    f.write(latex_perf)
print("\n✓ 已保存: train200_performance_table.tex (LaTeX)")

# ============== 保存原始数据 ==============
# 保存为JSON方便后续使用
import json
full_data = {
    'training_data': data,
    'validation_data': eval_data,
    'configuration': {
        'model': 'GPT-2 (124M)',
        'method': 'LoRA',
        'rank': 8,
        'alpha': 16,
        'steps': 200,
        'batch_size': 4,
        'grad_accum': 2,
        'seq_len': 128,
        'lr_max': 3e-4,
        'warmup_steps': 50,
        'dataset': 'WikiText-2',
    },
    'results': {
        'initial_valid_ppl': 44.54,
        'final_valid_ppl': 22.26,
        'improvement_percent': 50.0,
        'initial_train_loss': 3.54,
        'final_train_loss': 3.07,
        'final_ema_loss': 3.06,
    }
}

with open('/Users/yiyilu/Desktop/FT/train200_data.json', 'w') as f:
    json.dump(full_data, f, indent=2)
print("\n✓ 已保存: train200_data.json (原始数据)")

print("\n" + "="*80)
print("✅ 所有图表和表格生成完成！")
print("="*80)
print("\n生成的文件列表：")
print("  图表 (适合论文):")
print("    - train200_comprehensive.pdf/png     (3合1综合图)")
print("    - train200_loss_curve.pdf/png         (Loss曲线)")
print("    - train200_perplexity.pdf/png         (PPL曲线)")
print("    - train200_lr_schedule.pdf/png        (学习率调度)")
print("\n  表格 (LaTeX格式):")
print("    - train200_summary_table.tex          (配置摘要)")
print("    - train200_eval_table.tex             (验证集结果)")
print("    - train200_train_table.tex            (训练loss)")
print("    - train200_performance_table.tex      (性能统计)")
print("\n  数据文件:")
print("    - train200_detailed.csv               (完整训练数据)")
print("    - train200_data.json                  (原始数据)")
print("="*80)

