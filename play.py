import os
import sys
import time
import random
import numpy as np
import torch
import math
import argparse

from game import StayPositiveGame, get_card_name, get_card_suit, get_card_value, get_card_effective_value, calculate_turn_score
from env import StayPositiveEnv
from agent import DQNAgent
from train import select_greedy_action, select_random_action

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_action_mask(game, player_idx):
    mask = np.zeros(54, dtype=bool)
    for card in game.hands[player_idx]:
        mask[card] = True
    return mask

def print_game_header(game, user_player_idx, dqn_player_idx, roles):
    print("=" * 60)
    print(" " * 20 + "STAY POSITIVE CARD GAME")
    print("=" * 60)
    print(f"Top Card on Pile: {get_card_name(game.top_card) if game.top_card is not None else 'EMPTY'}")
    print("-" * 60)
    print(f"{'Player':<12} | {'Role':<15} | {'Score':<8} | {'Cards Left':<10}")
    print("-" * 60)
    for p in range(game.num_players):
        role_str = roles[p]
        if p == user_player_idx:
            role_str += " (You)"
        elif p == dqn_player_idx:
            role_str += " (DQN)"
        
        score_str = f"{game.scores[p]:.1f}"
        cards_left = len(game.hands[p])
        
        # Highlight active player
        if p == game.current_player:
            print(f"-> Player {p:<9} | {role_str:<15} | {score_str:<8} | {cards_left:<10}")
        else:
            print(f"   Player {p:<9} | {role_str:<15} | {score_str:<8} | {cards_left:<10}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Play Stay Positive Card Game interactively")
    parser.add_argument("--diamond-zero", "--diamond-is-zero", dest="diamond_is_zero", action="store_true",
                        help="Enable rule where effective value of Diamond cards is 0")
    args = parser.parse_args()

    clear_screen()
    print("Welcome to Stay Positive Interactive Play!")
    
    # 1. Detect and choose DQN checkpoint
    workspace_files = os.listdir('.')
    chk_files = sorted([f for f in workspace_files if f.startswith('stay_positive_dqn') and f.endswith('.pth')])
    
    if not chk_files:
        print("No DQN weight files (stay_positive_dqn*.pth) found in the current directory.")
        print("Please run train.py first to generate the models, or make sure the file is placed here.")
        sys.exit(1)
        
    print("\nAvailable DQN models:")
    for idx, f in enumerate(chk_files):
        # Infer training completion percentage if format allows
        pct_str = ""
        if "_0.pth" in f:
            pct_str = " (0% trained - Untrained)"
        elif "_3000.pth" in f:
            pct_str = " (25% trained)"
        elif "_6000.pth" in f:
            pct_str = " (50% trained)"
        elif "_9000.pth" in f:
            pct_str = " (75% trained)"
        elif "_12000.pth" in f:
            pct_str = " (100% trained)"
        elif f == "stay_positive_dqn.pth":
            pct_str = " (Final model)"
        print(f"  [{idx + 1}] {f}{pct_str}")
        
    while True:
        try:
            choice = input(f"Select a model to play against [1-{len(chk_files)}]: ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(chk_files):
                selected_model_path = chk_files[choice_idx]
                break
        except ValueError:
            pass
        print("Invalid choice. Please select a valid number.")
        
    # Rule toggle prompt if not explicitly passed in command line arguments
    diamond_is_zero = args.diamond_is_zero
    if not diamond_is_zero:
        dz_input = input("\nEnable rule where effective value of Diamond card is 0? [y/N]: ").strip().lower()
        if dz_input in ["y", "yes"]:
            diamond_is_zero = True
            
    # Load DQNAgent 
    env = StayPositiveEnv(diamond_is_zero=diamond_is_zero)
    dqn_state_dim = env.observation_space_size 
    dqn_agent = DQNAgent(state_dim=dqn_state_dim, action_dim=54)
    dqn_agent.load(selected_model_path)
    print(f"\nLoaded DQN model from {selected_model_path} (Diamond=0 Rule: {diamond_is_zero})")
    
    # 2. Number of bots (default 4, max total players 9)
    # total_players = 1 (user) + 1 (dqn) + num_bots
    # Therefore, num_bots can range from 4 to 7
    while True:
        try:
            bots_input = input("\nEnter number of bots to add [4-7] (default 4): ").strip()
            if bots_input == "":
                num_bots = 4
                break
            num_bots = int(bots_input)
            if 4 <= num_bots <= 7:
                break
        except ValueError:
            pass
        print("Invalid input. Please enter an integer between 4 and 7.")
        
    num_players = 2 + num_bots
    print(f"Total players in game: {num_players} (You, DQN, and {num_bots} bot(s))")
    
    # 3. Bot behaviors (Greedy, Random, Mixed)
    bot_style = "Greedy"
    if num_bots > 0:
        while True:
            style_input = input("Select bot behavior (G: Greedy [default], R: Random, M: Mixed): ").strip().upper()
            if style_input in ["", "G", "GREEDY"]:
                bot_style = "Greedy"
                break
            elif style_input in ["R", "RANDOM"]:
                bot_style = "Random"
                break
            elif style_input in ["M", "MIXED"]:
                bot_style = "Mixed"
                break
            print("Invalid input.")
            
    # 4. User player position
    while True:
        pos_input = input(f"Select your player index [0-{num_players-1}] (or R for Random) [default R]: ").strip().upper()
        if pos_input == "" or pos_input == "R":
            user_player_idx = random.randint(0, num_players - 1)
            break
        try:
            val = int(pos_input)
            if 0 <= val < num_players:
                user_player_idx = val
                break
        except ValueError:
            pass
        print(f"Invalid position. Choose between 0 and {num_players-1} or R.")
        
    # Assign DQN player index (must be different from user)
    available_indices = [i for i in range(num_players) if i != user_player_idx]
    dqn_player_idx = random.choice(available_indices)
    
    # Assign Roles
    roles = {}
    bot_counter = 1
    for p in range(num_players):
        if p == user_player_idx:
            roles[p] = "User"
        elif p == dqn_player_idx:
            roles[p] = "DQN Agent"
        else:
            # Determine bot type
            if bot_style == "Greedy":
                roles[p] = "Greedy Bot"
            elif bot_style == "Random":
                roles[p] = "Random Bot"
            else:
                # Mixed: alternate greedy and random
                roles[p] = "Greedy Bot" if bot_counter % 2 == 1 else "Random Bot"
                bot_counter += 1
                
    print(f"\nRoles assigned:")
    for p in range(num_players):
        spec = " (You)" if p == user_player_idx else " (DQN)" if p == dqn_player_idx else ""
        print(f"  Player {p}: {roles[p]}{spec}")
        
    input("\nPress Enter to start the game...")
    
    # Initialize game
    game_idx = num_players - env.min_players
    env.current_game_ind = game_idx
    game = env.games[game_idx]
    dqn_decisions = {"top_cards": [], "hands": [], "choices": []}
    
    # Game Loop
    while not game.is_game_over():
        clear_screen()
        print_game_header(game, user_player_idx, dqn_player_idx, roles)
        
        # Print recent history
        if game.history:
            last_turn = game.history[-1]
            last_p = last_turn['player']
            last_role = roles[last_p]
            print(f"Last Play: Player {last_p} ({last_role}) played {last_turn['card_name']}.")
            print(f"           Turn Score: {last_turn['turn_score']:.1f}")
            print("-" * 60)
            
        current_p = game.current_player
        
        if current_p == user_player_idx:
            # User turn
            print("YOUR TURN! Here is your hand:")
            hand = game.hands[user_player_idx]
            for idx, card in enumerate(hand):
                eff = get_card_effective_value(card, game.diamond_is_zero)
                eff_str = f"eff={eff:.2f}"
                pred = calculate_turn_score(game.top_card, card, game.diamond_is_zero)
                eff_str += f", projected score={pred:.0f}"
                print(f"  [{idx + 1}] {get_card_name(card)} ({eff_str})")
                
            # Ask for input
            while True:
                try:
                    user_select = input(f"\nChoose card to play [1-{len(hand)}]: ").strip()
                    card_idx_in_hand = int(user_select) - 1
                    if 0 <= card_idx_in_hand < len(hand):
                        action = hand[card_idx_in_hand]
                        break
                except ValueError:
                    pass
                print("Invalid input. Please enter a valid number.")
                
        elif current_p == dqn_player_idx:
            # DQN turn
            dqn_decisions["top_cards"].append(get_card_name(game.top_card))
            hand = game.hands[current_p]
            dqn_decisions["hands"].append(str(hand))
            print("DQN is thinking...")
            obs = env._get_observation(dqn_player_idx)
            mask = get_action_mask(game, dqn_player_idx)
            action = dqn_agent.select_action(obs, mask, epsilon=0.0)
            dqn_decisions["choices"].append(get_card_name(action))
            print(f"DQN selected: {get_card_name(action)}")
            time.sleep(1.5)
            
        else:
            # Bot turn
            bot_role = roles[current_p]
            print(f"Player {current_p} ({bot_role}) is playing...")
            mask = get_action_mask(game, current_p)
            if "Greedy" in bot_role:
                action = select_greedy_action(game, current_p)
            else:
                action = select_random_action(mask)
            print(f"{bot_role} selected: {get_card_name(action)}")
            time.sleep(1.2)
            
        # Play the card
        game.play_card(action)
        
    # Game Over
    clear_screen()
    print("=" * 60)
    print(" " * 22 + "GAME OVER!")
    print("=" * 60)
    print("\nFinal Scores:")
    for p in range(game.num_players):
        role_str = roles[p]
        if p == user_player_idx:
            role_str += " (You)"
        elif p == dqn_player_idx:
            role_str += " (DQN)"
        print(f"  Player {p} ({role_str}): {game.scores[p]:.1f}")
        
    winners = game.get_winners()
    print("\nWinner(s):")
    for w in winners:
        role_str = roles[w]
        if w == user_player_idx:
            role_str += " (You)"
        elif w == dqn_player_idx:
            role_str += " (DQN)"
        print(f"  Player {w} ({role_str}) with score {game.scores[w]:.1f}!")
    print("=" * 60)

    print("\n\n\n")
    print(dqn_decisions)

if __name__ == "__main__":
    main()
