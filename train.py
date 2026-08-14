import numpy as np
import torch
import random
import os
from env import StayPositiveEnv
from agent import DQNAgent
from game import get_card_suit, get_card_value, get_card_effective_value, calculate_turn_score
import math
import argparse

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

def run_evaluation(agent, num_games=100, num_players=6, diamond_is_zero=False):
    """
    Evaluates the DQN agent against:
    - Player 0: DQN
    - Player 1: Greedy
    - Player 2: Random
    """
    env = StayPositiveEnv(diamond_is_zero=diamond_is_zero)
    game_ind = num_players - env.min_players
    num_dqn_agents = float(num_players/3)
    num_greedy_bots = float(math.ceil((num_players - num_dqn_agents)/2.0))
    num_random_bots = float(num_players) - num_greedy_bots - num_dqn_agents
    active_game = env.games[game_ind]
    win_counts = [0] * 3
    scores_sum = [0.0] * 3
    play_order = [0, 1, 2]

    for _ in range(num_games):
        obs, active_player, mask, done = env.reset()
        random.shuffle(play_order)
        
        while not done:
            if active_player % 3 == play_order[0]:
                # DQN evaluation (no exploration)
                action = agent.select_action(obs, mask, epsilon=0.0)
            elif active_player % 3 == play_order[1]:
                action = select_greedy_action(active_game, active_player)
            else:
                action = select_random_action(mask)
                
            obs, active_player, rewards, done, mask = env.step(action)
            
        winners = active_game.get_winners()
        for w in winners:
            if w % 3 == play_order[0]:  # DQN won
                win_counts[0] += 1
            elif w % 3 == play_order[1]:  # Greedy bot won
                win_counts[1] += 1
            else:
                win_counts[2] += 1  # Random bot won
        for p in range(num_players):
            if p % 3 == play_order[0]:
                scores_sum[0] += active_game.scores[p]/num_dqn_agents
            elif p % 3 == play_order[1]:
                scores_sum[1] += active_game.scores[p]/num_greedy_bots
            else:
                scores_sum[2] += active_game.scores[p]/num_random_bots
            
    win_rates = [win_counts[p] / num_games for p in range(3)]
    avg_scores = [scores_sum[p] / num_games for p in range(3)]
    return win_rates, avg_scores

def main():
    parser = argparse.ArgumentParser(description="Train DQN Agent for Stay Positive Card Game")
    parser.add_argument("--diamond-zero", "--diamond-is-zero", dest="diamond_is_zero", action="store_true",
                        help="Enable rule where effective value of Diamond cards is 0")
    parser.add_argument("--episodes", type=int, default=12000, help="Number of training episodes")
    parser.add_argument("--save-path", type=str, default="stay_positive_dqn.pth", help="Model output file path")
    args = parser.parse_args()

    num_episodes = args.episodes
    eval_interval = 1000
    save_path = args.save_path
    diamond_is_zero = args.diamond_is_zero
    
    # Initialize environment and agent
    env = StayPositiveEnv(diamond_is_zero=diamond_is_zero)
    state_dim = env.observation_space_size
    agent = DQNAgent(state_dim=state_dim, action_dim=54, lr=1e-4)
    
    # Checkpoint configuration
    checkpoint_increments = [0, 0.25, 0.5, 0.75, 1.0]
    checkpoint_episodes = [int(pct * num_episodes) for pct in checkpoint_increments]
    
    # Save 0% checkpoint (initial untrained state)
    checkpoint_0_path = save_path.replace(".pth", "_0.pth")
    agent.save(checkpoint_0_path)
    print(f"Saved initial untrained model checkpoint to {checkpoint_0_path}")
    
    # Exploration parameters
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_episodes = 9000
    
    print(f"Starting training on device: {agent.device} (Diamond=0 Rule: {diamond_is_zero})")
    print(f"State Dim: {state_dim}, Action Dim: 54")
    
    losses = []
    
    for episode in range(1, num_episodes + 1):
        # Determine exploration rate
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * min(1.0, episode / epsilon_decay_episodes)
        
        # Reset env
        obs, active_player, mask, done = env.reset()
        active_game = env.games[env.current_game_ind]
        num_players = active_game.num_players
        
        # Track history for all players to construct transitions
        state_history = {p: None for p in range(num_players)}
        action_history = {p: None for p in range(num_players)}
        mask_history = {p: None for p in range(num_players)}
        
        # Determine opponent types for this episode to ensure balanced training
        # We want to make sure the agent trains against both random and greedy agents.
        opponent_types = {}
        if num_players >= 3:
            r_scen = random.random()
            if r_scen < 0.4:
                # 40% of games: DQN vs 1 Greedy vs 1 Random (and rest mixed if num_players > 3)
                opponent_types[1] = "Greedy"
                opponent_types[2] = "Random"
                for p in range(3, num_players):
                    opponent_types[p] = random.choice(["DQN", "Greedy", "Random"])
            elif r_scen < 0.6:
                # 20% of games: DQN vs all Greedy
                for p in range(1, num_players):
                    opponent_types[p] = "Greedy"
            elif r_scen < 0.8:
                # 20% of games: DQN vs all Random
                for p in range(1, num_players):
                    opponent_types[p] = "Random"
            else:
                # 20% of games: Self-play / Mixed opponents
                for p in range(1, num_players):
                    opponent_types[p] = random.choice(["DQN", "Greedy", "Random"])
        else:
            # 2 players: 50% Greedy, 50% Random
            for p in range(1, num_players):
                opponent_types[p] = "Greedy" if random.random() < 0.5 else "Random"
                
        # Run episode
        while not done:
            # Check if this player has a pending transition to save
            # The next state for their previous action is the current 'obs'
            if state_history[active_player] is not None:
                # Only push to buffer if this player was using the DQN policy
                if active_player == 0 or opponent_types.get(active_player) == "DQN":
                    agent.memory.push(
                        state_history[active_player],
                        action_history[active_player],
                        0.0, # intermediate reward
                        obs,
                        False, # not done
                        mask # next action mask
                    )
            
            # Select action
            if active_player == 0 or opponent_types.get(active_player) == "DQN":
                action = agent.select_action(obs, mask, epsilon)
            elif opponent_types.get(active_player) == "Greedy":
                action = select_greedy_action(active_game, active_player)
            else:
                action = select_random_action(mask)
                
            # Store current state-action
            state_history[active_player] = obs
            action_history[active_player] = action
            mask_history[active_player] = mask
            
            # Step game
            obs, active_player, rewards, done, mask = env.step(action)
            
            # Update network parameters
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
                
        # Game is done. Push terminal transitions for all DQN players.
        for p in range(num_players):
            if state_history[p] is not None:
                if p == 0 or opponent_types.get(p) == "DQN":
                    agent.memory.push(
                        state_history[p],
                        action_history[p],
                        rewards[p], # terminal reward (+1.0 or -1.0)
                        np.zeros_like(state_history[p]),
                        True,
                        np.zeros(54, dtype=bool) # terminal state mask is all False
                    )
                    
        # Update DQN training step one last time for the episode
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)
            
        # Logging & Evaluation
        if episode % eval_interval == 0:
            win_rates, avg_scores = run_evaluation(agent, num_games=100, diamond_is_zero=diamond_is_zero)
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"Episode {episode}/{num_episodes} | Epsilon: {epsilon:.3f} | Avg Loss: {avg_loss:.4f}")
            print(f"  Win Rates: DQN: {win_rates[0]:.2f} | Greedy: {win_rates[1]:.2f} | Random: {win_rates[2]:.2f}")
            print(f"  Avg Scores: DQN: {avg_scores[0]:.1f} | Greedy: {avg_scores[1]:.1f} | Random: {avg_scores[2]:.1f}")
            
        # Save checkpoints at episode increments
        if episode in checkpoint_episodes:
            checkpoint_path = save_path.replace(".pth", f"_{episode}.pth")
            agent.save(checkpoint_path)
            print(f"Saved checkpoint for episode {episode} to {checkpoint_path}")

        # Switch number of players in game
        env.current_game_ind = (env.current_game_ind + 1) % len(env.games)
            
    # Save the final agent
    agent.save(save_path)
    print(f"Saved trained DQN model to {save_path}")

if __name__ == "__main__":
    main()
