import numpy as np
import torch
import random
import os
import math
import argparse

from env import StayPositiveEnv
from agent import DQNAgent
from game import get_card_suit, get_card_value, get_card_effective_value, calculate_turn_score

# Baseline Agent Selectors
def select_random_action(action_mask):
    valid_indices = np.where(action_mask)[0]
    return random.choice(valid_indices)

def select_greedy_action(game, player_idx):
    valid_cards = game.hands[player_idx]
    if not valid_cards:
        return 0
    
    best_score = -float('inf')
    best_cards = []
    diamond_is_zero = getattr(game, 'diamond_is_zero', False)
    
    for card in valid_cards:
        turn_score = calculate_turn_score(game.top_card, card, diamond_is_zero)
        if turn_score > best_score:
            best_score = turn_score
            best_cards = [card]
        elif turn_score == best_score:
            best_cards.append(card)
            
    return random.choice(best_cards)

def get_player_roles(num_players):
    """
    Returns the player composition list for each game size:
    - 6 players: 2 DQN agents, 2 Random, 2 Greedy
    - 7 players: 2 DQN agents, 3 Random, 2 Greedy
    - 8 players: 2 DQN agents, 3 Random, 3 Greedy
    - 9 players: 3 DQN agents, 3 Random, 3 Greedy
    Each tuple is (role_type, dqn_agent_index).
    """
    if num_players == 6:
        return [("DQN", 0), ("DQN", 1), ("Greedy", None), ("Greedy", None), ("Random", None), ("Random", None)]
    elif num_players == 7:
        return [("DQN", 0), ("DQN", 1), ("Greedy", None), ("Greedy", None), ("Random", None), ("Random", None), ("Random", None)]
    elif num_players == 8:
        return [("DQN", 0), ("DQN", 1), ("Greedy", None), ("Greedy", None), ("Greedy", None), ("Random", None), ("Random", None), ("Random", None)]
    elif num_players == 9:
        return [("DQN", 0), ("DQN", 1), ("DQN", 2), ("Greedy", None), ("Greedy", None), ("Greedy", None), ("Random", None), ("Random", None), ("Random", None)]
    else:
        raise ValueError(f"Unsupported number of players: {num_players}")

def run_evaluation(dqn_agents, num_games=100, diamond_is_zero=False):
    """
    Evaluates the DQN agents exclusively in 6-player games:
    - 2 DQN agents (dqn_agents[0] and dqn_agents[1])
    - 2 Greedy bots
    - 2 Random bots
    Player play order and starting player are randomized for each game.
    """
    num_players = 6
    env = StayPositiveEnv(min_players=6, max_players=6, diamond_is_zero=diamond_is_zero)
    base_roles = [("DQN", 0), ("DQN", 1), ("Greedy", None), ("Greedy", None), ("Random", None), ("Random", None)]
    
    win_counts = [0] * 3  # [DQN, Greedy, Random]
    scores_sum = [0.0] * 3  # [DQN, Greedy, Random]

    for _ in range(num_games):
        obs, active_player, mask, done = env.reset()
        active_game = env.games[env.current_game_ind]
        
        # Randomize play order / role assignment for the 6 players
        player_roles = list(base_roles)
        random.shuffle(player_roles)

        while not done:
            role_type, dqn_idx = player_roles[active_player]
            
            if role_type == "DQN":
                action = dqn_agents[dqn_idx].select_action(obs, mask, epsilon=0.0)
            elif role_type == "Greedy":
                action = select_greedy_action(active_game, active_player)
            else:
                action = select_random_action(mask)
                
            obs, active_player, rewards, done, mask = env.step(action)
            
        winners = active_game.get_winners()
        for w in winners:
            w_role, _ = player_roles[w]
            if w_role == "DQN":
                win_counts[0] += 1
            elif w_role == "Greedy":
                win_counts[1] += 1
            else:
                win_counts[2] += 1
                
        for p in range(num_players):
            p_role, _ = player_roles[p]
            if p_role == "DQN":
                scores_sum[0] += active_game.scores[p] / 2.0
            elif p_role == "Greedy":
                scores_sum[1] += active_game.scores[p] / 2.0
            else:
                scores_sum[2] += active_game.scores[p] / 2.0
            
    win_rates = [win_counts[p] / num_games for p in range(3)]
    avg_scores = [scores_sum[p] / num_games for p in range(3)]
    return win_rates, avg_scores

def main():
    parser = argparse.ArgumentParser(description="Train Multi-DQN Agents for Stay Positive Card Game")
    parser.add_argument("--diamond-zero", "--diamond-is-zero", dest="diamond_is_zero", action="store_true",
                        help="Enable rule where effective value of Diamond cards is 0")
    parser.add_argument("--episodes", type=int, default=12000, help="Number of training episodes")
    parser.add_argument("--save-path", type=str, default="stay_positive_dqn.pth", help="Model output file path")
    args = parser.parse_args()

    num_episodes = args.episodes
    eval_interval = 1000
    save_path = args.save_path
    diamond_is_zero = args.diamond_is_zero
    
    # Initialize environment and 3 separate DQN agents
    env = StayPositiveEnv(min_players=6, max_players=9, diamond_is_zero=diamond_is_zero)
    state_dim = env.observation_space_size
    
    dqn_agents = [DQNAgent(state_dim=state_dim, action_dim=54, lr=1e-4) for _ in range(3)]
    
    # Checkpoint configuration
    checkpoint_increments = [0, 0.25, 0.5, 0.75, 1.0]
    checkpoint_episodes = [int(pct * num_episodes) for pct in checkpoint_increments]
    
    # Save 0% checkpoint (initial untrained state for primary agent)
    checkpoint_0_path = save_path.replace(".pth", "_0.pth")
    dqn_agents[0].save(checkpoint_0_path)
    print(f"Saved initial untrained model checkpoint to {checkpoint_0_path}")
    
    # Exploration parameters
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_episodes = 9000
    
    print(f"Starting training on device: {dqn_agents[0].device} (Diamond=0 Rule: {diamond_is_zero})")
    print(f"State Dim: {state_dim}, Action Dim: 54, Number of DQN Models: 3")
    
    losses = []
    
    for episode in range(1, num_episodes + 1):
        # Determine exploration rate
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * min(1.0, episode / epsilon_decay_episodes)
        
        # Reset env
        obs, active_player, mask, done = env.reset()
        active_game = env.games[env.current_game_ind]
        num_players = active_game.num_players
        
        # Get player roles setup for this game size and randomize play order / role assignment
        player_roles = get_player_roles(num_players)
        random.shuffle(player_roles)
        
        # Track history for all players to construct transitions
        state_history = {p: None for p in range(num_players)}
        action_history = {p: None for p in range(num_players)}
        mask_history = {p: None for p in range(num_players)}
        
        # Run episode
        while not done:
            role_type, dqn_idx = player_roles[active_player]
            
            # If active player is a DQN agent and has a pending transition, save it
            if state_history[active_player] is not None and role_type == "DQN":
                agent = dqn_agents[dqn_idx]
                agent.memory.push(
                    state_history[active_player],
                    action_history[active_player],
                    0.0,  # intermediate reward
                    obs,
                    False,  # not done
                    mask  # next action mask
                )
            
            # Select action
            if role_type == "DQN":
                agent = dqn_agents[dqn_idx]
                action = agent.select_action(obs, mask, epsilon)
            elif role_type == "Greedy":
                action = select_greedy_action(active_game, active_player)
            else:
                action = select_random_action(mask)
                
            # Store current state-action
            state_history[active_player] = obs
            action_history[active_player] = action
            mask_history[active_player] = mask
            
            # Step game
            obs, active_player, rewards, done, mask = env.step(action)
            
            # Update network parameters for the specific DQN model that just moved
            if role_type == "DQN":
                agent = dqn_agents[dqn_idx]
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
                
        # Game is done. Push terminal transitions for all DQN players in this game.
        for p in range(num_players):
            role_type, dqn_idx = player_roles[p]
            if role_type == "DQN" and state_history[p] is not None:
                agent = dqn_agents[dqn_idx]
                agent.memory.push(
                    state_history[p],
                    action_history[p],
                    rewards[p],  # terminal reward (+1.0 or -1.0)
                    np.zeros_like(state_history[p]),
                    True,
                    np.zeros(54, dtype=bool)  # terminal state mask is all False
                )
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
            
        # Logging & Evaluation (in 6-player games)
        if episode % eval_interval == 0:
            win_rates, avg_scores = run_evaluation(dqn_agents, num_games=100, diamond_is_zero=diamond_is_zero)
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"Episode {episode}/{num_episodes} | Epsilon: {epsilon:.3f} | Avg Loss: {avg_loss:.4f}")
            print(f"  [6-Player Evaluation] Win Rates: DQN: {win_rates[0]:.2f} | Greedy: {win_rates[1]:.2f} | Random: {win_rates[2]:.2f}")
            print(f"                        Avg Scores: DQN: {avg_scores[0]:.1f} | Greedy: {avg_scores[1]:.1f} | Random: {avg_scores[2]:.1f}")
            
        # Save checkpoints at episode increments
        if episode in checkpoint_episodes:
            checkpoint_path = save_path.replace(".pth", f"_{episode}.pth")
            dqn_agents[0].save(checkpoint_path)
            print(f"Saved checkpoint for episode {episode} to {checkpoint_path}")

        # Switch number of players in game
        env.current_game_ind = (env.current_game_ind + 1) % len(env.games)
            
    # Save the trained DQN models
    dqn_agents[0].save(save_path)
    for i in range(3):
        dqn_agents[i].save(f"stay_positive_dqn_agent{i}.pth")
    print(f"Saved trained DQN models to {save_path} and stay_positive_dqn_agent[0-2].pth")

if __name__ == "__main__":
    main()
