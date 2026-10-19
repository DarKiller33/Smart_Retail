import pandas as pd
import matplotlib.pyplot as plt
import os

# Set publication style
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 12
})

def plot_research_figures(csv_file="edge_metrics_log.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: '{csv_file}' not found. Please run main_pipeline.py first to generate telemetry logs.")
        return

    df = pd.read_csv(csv_file)
    os.makedirs("paper_plots", exist_ok=True)

    # -------------------------------------------------------------
    # Figure 1: Pipeline Latency & FPS Stability
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(7, 3.5), dpi=300)

    color_latency = '#d95f02'
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Inference Latency (ms)', color=color_latency)
    ax1.plot(df['Frame'], df['Inference_Time_ms'], color=color_latency, linewidth=1.2, label='Inference Latency')
    ax1.tick_params(axis='y', labelcolor=color_latency)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color_fps = '#1b9e77'
    ax2.set_ylabel('Throughput (FPS)', color=color_fps)
    ax2.plot(df['Frame'], df['FPS'], color=color_fps, linewidth=1.2, linestyle=':', label='Throughput (FPS)')
    ax2.tick_params(axis='y', labelcolor=color_fps)

    plt.title("Real-Time Latency and Throughput Stability", pad=12)
    fig.tight_layout()
    plt.savefig("paper_plots/fig1_latency_fps_stability.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Edge Resource Consumption (CPU % & RAM %)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)

    ax.plot(df['Frame'], df['CPU_Util_Percent'], label='CPU Utilization (%)', color='#7570b3', linewidth=1.2)
    ax.plot(df['Frame'], df['RAM_Util_Percent'], label='RAM Overhead (%)', color='#e7298a', linewidth=1.2, linestyle='--')

    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Utilization Percentage (%)')
    ax.set_title("Edge Hardware Resource Footprint", pad=12)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig("paper_plots/fig2_hardware_utilization.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Figure 3: Execution Engine Benchmark Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)

    engines = ['PyTorch FP32\n(Baseline)', 'OpenVINO\n(Edge Optimized)']
    latencies = [75.72, 17.66]
    colors = ['#e41a1c', '#377eb8']

    bars = ax.bar(engines, latencies, color=colors, width=0.45, edgecolor='black', linewidth=0.8)
    
    # Add numerical labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.2f} ms", ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Average Latency per Frame (ms)')
    ax.set_title("Inference Engine Latency Comparison", pad=12)
    ax.set_ylim(0, 95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.savefig("paper_plots/fig3_engine_comparison.png", dpi=300)
    plt.close()

    print("Success! High-resolution research plots saved inside the 'paper_plots/' directory.")

if __name__ == "__main__":
    plot_research_figures()