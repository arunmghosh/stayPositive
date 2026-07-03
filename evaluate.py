import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os
from env import StayPositiveEnv
from agent import DQNAgent
from train import select_greedy_action, select_random_action

def evaluate_agents(agent_path, num_games=1000, num_players=6, artifact_dir=None):
    env = StayPositiveEnv()
    state_dim = env.observation_space_size
    
    # Load DQN Agent
    dqn_agent = DQNAgent(state_dim=state_dim, action_dim=54)
    if os.path.exists(agent_path):
        dqn_agent.load(agent_path)
        print(f"Loaded trained agent from {agent_path}")
    else:
        print(f"Warning: {agent_path} not found! Evaluating an untrained agent.")

    # Tracking metrics
    # Agent types: 'DQN', 'Greedy', 'Random'
    agent_types = ['DQN', 'Greedy', 'Random']
    
    wins = {t: 0 for t in agent_types}
    scores = {t: [] for t in agent_types}
    
    # Track wins by which agent type started
    # play_order_wins[agent_type][start_type_ind] = # of wins agent_type got when start_type started the game
    # play_order_games[start_type] = # of games in which start_type started the game
    play_order_wins = {t: [0, 0, 0] for t in agent_types}
    play_order_games = {t: 0 for t in agent_types}

    for game_idx in range(num_games):
        obs, active_player, mask, done = env.reset()
        active_game = env.games[env.current_game_ind]
        
        # Assign agent types to players randomly
        # player_mapping[player_idx] = agent_type
        shuffled_agents = list()
        for p in range(num_players):
            shuffled_agents.append(agent_types[p % len(agent_types)])
        random.shuffle(shuffled_agents)
        player_mapping = {p: shuffled_agents[p] for p in range(num_players)}
        
        # Identify starting order
        # starting_player plays 1st.
        start_p = active_game.starting_player
        play_order_games[player_mapping[start_p]] += 1

        order = list()
        order.append(start_p)
        for o in range(1, num_players):
            order.append((start_p + o) % num_players)
            
        while not done:
            agent_type = player_mapping[active_player]
            
            if agent_type == 'DQN':
                action = dqn_agent.select_action(obs, mask, epsilon=0.0)
            elif agent_type == 'Greedy':
                action = select_greedy_action(active_game, active_player)
            else:
                action = select_random_action(mask)
                
            obs, active_player, rewards, done, mask = env.step(action)
            
        # Determine winners
        winners = active_game.get_winners()
        for w in winners:
            win_agent = player_mapping[w]
            wins[win_agent] += 1
            
            # Update play_order_wins
            play_order_wins[win_agent][agent_types.index(player_mapping[start_p])] += 1
            
        # Record scores
        for p in range(num_players):
            agent_type = player_mapping[p]
            scores[agent_type].append(active_game.scores[p])
            
    # Calculate statistics
    win_rates = {t: wins[t] / num_games for t in agent_types}
    avg_scores = {t: np.mean(scores[t]) for t in agent_types}
    std_scores = {t: np.std(scores[t]) for t in agent_types}
    
    print("\n================ EVALUATION RESULTS ================")
    print(f"Total Games Played: {num_games}")
    print(f"{'Agent Type':<12} | {'Win Rate':<10} | {'Avg Score (Std)':<15}")
    print("-" * 50)
    for t in agent_types:
        print(f"{t:<12} | {win_rates[t]:.2%}   | {avg_scores[t]:.1f} (+/- {std_scores[t]:.1f})")
    print("====================================================")
    
    print("\nWin Rate by Starting Type:")
    for t in agent_types:
        start_type_win_rates = []
        for st in range(3):  # cycles through start types
            games_played = play_order_games[agent_types[st]]
            wins_got = play_order_wins[t][st]
            rate = wins_got / games_played if games_played > 0 else 0.0
            start_type_win_rates.append(f"{agent_types[st]}: {rate:.1%}")
        print(f"  {t:<8}: {', '.join(start_type_win_rates)}")
        
    # Generate Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Win Rate comparison
    ax1.bar(agent_types, [win_rates[t] * 100 for t in agent_types], color=['#4285F4', '#EA4335', '#FBBC05'])
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Stay Positive: Win Rate Comparison')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Avg Score comparison
    ax2.bar(agent_types, [avg_scores[t] for t in agent_types], color=['#4285F4', '#EA4335', '#FBBC05'])
    ax2.set_ylabel('Average Cumulative Score')
    ax2.set_title('Stay Positive: Avg Final Scores')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if artifact_dir:
        plot_path = os.path.join(artifact_dir, "evaluation_results.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Saved evaluation plots to {plot_path}")
        
    plt.close()

if __name__ == "__main__":
    evaluate_agents(
        agent_path="stay_positive_dqn.pth",
        num_games=1000,
        artifact_dir="/Users/arunghosh/.gemini/antigravity-ide/brain/5c0780a9-d05c-46f3-b7d0-7757b58bac85"
    )
