import os
import matplotlib.pyplot as plt
from env import StayPositiveEnv
from agent import DQNAgent
from train import run_evaluation

def evaluate_checkpoint(agent_path, num_games=200, num_players=6):
    env = StayPositiveEnv()
    state_dim = env.observation_space_size
    agent = DQNAgent(state_dim=state_dim, action_dim=54)
    
    if not os.path.exists(agent_path):
        print(f"Checkpoint {agent_path} not found. Skipping.")
        return None, None
        
    agent.load(agent_path)
    
    return run_evaluation(agent, num_games, num_players)

def main():
    checkpoints = {
        0: "stay_positive_dqn_0.pth",
        3000: "stay_positive_dqn_3000.pth",
        6000: "stay_positive_dqn_6000.pth",
        9000: "stay_positive_dqn_9000.pth",
        12000: "stay_positive_dqn_12000.pth"
    }
    
    # Fallback to base stay_positive_dqn.pth for 12000 if 12000.pth is not created yet
    if not os.path.exists("stay_positive_dqn_12000.pth") and os.path.exists("stay_positive_dqn.pth"):
        checkpoints[12000] = "stay_positive_dqn.pth"
        
    print("=" * 60)
    print("Evaluating DQN Checkpoints (DQN vs Greedy vs Random)")
    print("=" * 60)
    print(f"{'Episodes':<10} | {'DQN Win Rate':<12} | {'Greedy Win Rate':<15} | {'Random Win Rate':<15}")
    print("-" * 60)
    
    episodes_list = []
    dqn_win_rates = []
    greedy_win_rates = []
    random_win_rates = []
    
    dqn_scores = []
    greedy_scores = []
    
    for ep, path in sorted(checkpoints.items()):
        if not os.path.exists(path):
            continue
            
        win_rates, avg_scores = evaluate_checkpoint(path, num_games=200)
        if win_rates is None:
            continue
            
        episodes_list.append(ep)
        dqn_win_rates.append(win_rates[0])
        greedy_win_rates.append(win_rates[1])
        random_win_rates.append(win_rates[2])
        
        dqn_scores.append(avg_scores[0])
        greedy_scores.append(avg_scores[1])
        
        print(f"{ep:<10} | {win_rates[0]:.1%}        | {win_rates[1]:.1%}          | {win_rates[2]:.1%}")
        
    if len(episodes_list) > 1:
        # Generate plot
        plt.figure(figsize=(10, 6))
        plt.plot(episodes_list, [r * 100 for r in dqn_win_rates], marker='o', linewidth=2.5, label='DQN Agent', color='#4285F4')
        plt.plot(episodes_list, [r * 100 for r in greedy_win_rates], marker='s', linestyle='--', label='Greedy Baseline', color='#EA4335')
        plt.plot(episodes_list, [r * 100 for r in random_win_rates], marker='x', linestyle=':', label='Random Baseline', color='#FBBC05')
        
        plt.xlabel('Training Episodes', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.title('Stay Positive: DQN Strategy Discovery (Win Rate Progression)', fontsize=14, fontweight='bold')
        plt.xticks(episodes_list)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=11)
        
        # Save to artifact directory
        artifact_dir = "/Users/arunghosh/.gemini/antigravity-ide/brain/2e3a5d6e-45b9-491e-af8b-34f6d438c863"
        os.makedirs(artifact_dir, exist_ok=True)
        plot_path = os.path.join(artifact_dir, "checkpoints_evaluation.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nGenerated progression plot saved to {plot_path}")
    else:
        print("\nNot enough checkpoints found to plot progression.")

if __name__ == "__main__":
    main()
