import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_learning_curves():
    os.makedirs("plots", exist_ok=True)
    ppo_path = 'logs/ppo_v3/progress.csv'
    td3_path = 'logs/td3_v3/progress.csv'
    
    plt.figure(figsize=(12, 6))
    
    # 1. PPO Learning Curve
    if os.path.exists(ppo_path):
        ppo_df = pd.read_csv(ppo_path)
        plt.plot(ppo_df['time/total_timesteps'], ppo_df['rollout/ep_rew_mean'], label='PPO V3', color='blue', alpha=0.8)
    
    # 2. TD3 Learning Curve
    if os.path.exists(td3_path):
        td3_df = pd.read_csv(td3_path)
        plt.plot(td3_df['time/total_timesteps'], td3_df['rollout/ep_rew_mean'], label='TD3 V3', color='green', alpha=0.8)
    
    plt.title('QUBE-Servo 2: V3 Training Comparison (Learning Curves)')
    plt.xlabel('Total Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/v3_learning_curves.png', dpi=200)
    print('Learning curve plot saved to v3_learning_curves.png')

    # Also plot loss comparison if data exists
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    if os.path.exists(ppo_path):
        plt.plot(ppo_df['time/total_timesteps'], ppo_df['train/value_loss'], label='PPO Value Loss', color='red')
    plt.title('PPO Value Loss')
    plt.xlabel('Timesteps')
    plt.grid(True, alpha=0.2)
    
    plt.subplot(1, 2, 2)
    if os.path.exists(td3_path):
        plt.plot(td3_df['time/total_timesteps'], td3_df['train/critic_loss'], label='TD3 Critic Loss', color='purple')
    plt.title('TD3 Critic Loss')
    plt.xlabel('Timesteps')
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('plots/v3_loss_curves.png', dpi=200)
    print('Loss curve plot saved to v3_loss_curves.png')

if __name__ == "__main__":
    plot_learning_curves()
