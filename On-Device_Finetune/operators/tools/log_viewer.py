#!/usr/bin/env python3
"""
LoRA Fine-tuning Log Viewer
Tool for analyzing and visualizing GPT2 and Qwen LoRA fine-tuning training logs
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import seaborn as sns

def read_training_log(log_file):
    """Read training log file"""
    logs = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '[INFO]' in line and ('Loss=' in line or 'complete' in line or '完成' in line):
                logs.append(line.strip())
    return logs

def read_metrics_csv(csv_file):
    """Read metrics CSV file"""
    try:
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def plot_training_curves(df, output_dir="plots"):
    """Plot training curves"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set font settings
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Filter out EPOCH_END rows
    train_df = df[df['step'] != 'EPOCH_END'].copy()
    train_df['step'] = train_df['step'].astype(int)
    
    # 1. Loss curve
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for epoch in train_df['epoch'].unique():
        epoch_data = train_df[train_df['epoch'] == epoch]
        plt.plot(epoch_data['step'], epoch_data['loss'], 
                label=f'Epoch {epoch}', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss per Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Average loss curve
    plt.subplot(2, 2, 2)
    for epoch in train_df['epoch'].unique():
        epoch_data = train_df[train_df['epoch'] == epoch]
        plt.plot(epoch_data['step'], epoch_data['avg_loss'], 
                label=f'Epoch {epoch}', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Change')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Step time distribution
    plt.subplot(2, 2, 3)
    plt.hist(train_df['step_time_ms'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Step Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Step Time Distribution')
    plt.grid(True, alpha=0.3)
    
    # 4. Loss vs time
    plt.subplot(2, 2, 4)
    plt.plot(train_df['timestamp'], train_df['loss'], alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.title('Loss over Time')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

def analyze_metrics(df):
    """Analyze training metrics"""
    train_df = df[df['step'] != 'EPOCH_END'].copy()
    train_df['step'] = train_df['step'].astype(int)
    
    print("=== Training Metrics Analysis ===")
    print(f"Total training steps: {len(train_df)}")
    print(f"Training epochs: {train_df['epoch'].nunique()}")
    print(f"Final loss: {train_df['loss'].iloc[-1]:.6f}")
    print(f"Minimum loss: {train_df['loss'].min():.6f}")
    print(f"Average step time: {train_df['step_time_ms'].mean():.1f}ms")
    print(f"Slowest step time: {train_df['step_time_ms'].max():.1f}ms")
    
    # Statistics per epoch
    print("\n=== Per-Epoch Statistics ===")
    for epoch in sorted(train_df['epoch'].unique()):
        epoch_data = train_df[train_df['epoch'] == epoch]
        print(f"Epoch {epoch}:")
        print(f"  - Steps: {len(epoch_data)}")
        print(f"  - Average loss: {epoch_data['loss'].mean():.6f}")
        print(f"  - Final loss: {epoch_data['loss'].iloc[-1]:.6f}")
        print(f"  - Average step time: {epoch_data['step_time_ms'].mean():.1f}ms")

def main():
    parser = argparse.ArgumentParser(description='LoRA Fine-tuning Log Viewer')
    parser.add_argument('--log_dir', type=str, required=True, help='Log directory path')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--analyze', action='store_true', help='Analyze metrics')
    
    args = parser.parse_args()
    
    # Find metrics.csv files
    csv_files = []
    for file in os.listdir(args.log_dir):
        if file.endswith('_metrics.csv'):
            csv_files.append(os.path.join(args.log_dir, file))
    
    if not csv_files:
        print(f"No metrics CSV files found in {args.log_dir}")
        return
    
    for csv_file in csv_files:
        print(f"\nProcessing file: {csv_file}")
        df = read_metrics_csv(csv_file)
        
        if df is not None:
            if args.analyze:
                analyze_metrics(df)
            
            if args.plot:
                model_name = os.path.basename(csv_file).replace('_metrics.csv', '')
                output_dir = os.path.join(args.log_dir, f"{model_name}_plots")
                plot_training_curves(df, output_dir)
                print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
