import os
import sys
import random
import argparse
import numpy as np
import torch

from env import StayPositiveEnv
from agent import DQNAgent
from game import get_card_name, calculate_turn_score
from train import select_greedy_action, select_random_action, get_player_roles

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_role_label(role_type, dqn_idx):
    if role_type == "DQN":
        return f"DQN (agent{dqn_idx})"
    return role_type

def run_games_in_memory(num_games=10, diamond_is_zero=False):
    num_players = 9
    env = StayPositiveEnv(min_players=num_players, max_players=num_players, diamond_is_zero=diamond_is_zero)
    state_dim = env.observation_space_size

    # Load the 3 separate DQN agents (agent0, agent1, agent2)
    dqn_agents = []
    for i in range(3):
        agent = DQNAgent(state_dim=state_dim, action_dim=54)
        agent_path = f"stay_positive_dqn_agent{i}.pth"
        if not os.path.exists(agent_path):
            agent_path = "stay_positive_dqn.pth"
        
        if os.path.exists(agent_path):
            agent.load(agent_path)
            print(f"Loaded agent{i} model from {agent_path}")
        else:
            print(f"Warning: Model file {agent_path} not found for agent{i}! Using untrained model.")
        dqn_agents.append(agent)

    games_data = []

    print("\n" + "=" * 80)
    print(f"Simulating {num_games} 9-Player Games in Memory...")
    print(f"Rules: [Diamond=0: {diamond_is_zero}]")
    print("=" * 80)

    for game_idx in range(num_games):
        obs, active_player, mask, done = env.reset()
        active_game = env.games[env.current_game_ind]

        # Assign player roles for 9 players (3 DQN, 3 Greedy, 3 Random)
        player_roles = get_player_roles(num_players)
        random.shuffle(player_roles)
        role_labels = [get_role_label(r, idx) for r, idx in player_roles]

        moves = []
        turn_count = 0

        while not done:
            turn_count += 1
            curr_player = active_game.current_player
            role_type, dqn_idx = player_roles[curr_player]

            top_card_before = active_game.top_card
            top_card_name = get_card_name(top_card_before) if top_card_before is not None else "EMPTY"
            hand_before = list(active_game.hands[curr_player])
            hand_names = [get_card_name(c) for c in hand_before]

            if role_type == "DQN":
                action = int(dqn_agents[dqn_idx].select_action(obs, mask, epsilon=0.0))
            elif role_type == "Greedy":
                action = select_greedy_action(active_game, curr_player)
            else:
                action = select_random_action(mask)

            card_played_name = get_card_name(action)
            turn_score = calculate_turn_score(top_card_before, action, diamond_is_zero)
            
            obs, active_player, rewards, done, mask = env.step(action)
            scores_after = list(active_game.scores)

            moves.append({
                "move_num": turn_count,
                "player": curr_player,
                "role_label": role_labels[curr_player],
                "role_type": role_type,
                "dqn_idx": dqn_idx,
                "top_card_before": top_card_before,
                "top_card_name": top_card_name,
                "hand_before": hand_before,
                "hand_names": hand_names,
                "card_played": action,
                "card_played_name": card_played_name,
                "turn_score": turn_score,
                "scores_after": scores_after
            })

        games_data.append({
            "game_num": game_idx + 1,
            "starting_player": active_game.starting_player,
            "role_labels": role_labels,
            "moves": moves,
            "final_scores": [float(s) for s in active_game.scores],
            "winners": [int(w) for w in active_game.get_winners()]
        })

        print(f"  Game {game_idx + 1:2d}/{num_games} complete ({len(moves)} moves played).")

    print("\nAll 10 games finished playing in memory!")
    return games_data

def display_move(game_data, move_idx):
    moves = game_data["moves"]
    total_moves = len(moves)
    m = moves[move_idx]

    clear_screen()
    print("=" * 80)
    print(f" GAME {game_data['game_num']} OF {game_data['total_games']}  |  MOVE {m['move_num']} OF {total_moves}")
    print("=" * 80)
    print(f" Active Player : Player {m['player']} [{m['role_label']}]")
    print(f" Top Card      : {m['top_card_name']}")
    print(f" Card Played   : {m['card_played_name']}  (Turn Score: {m['turn_score']:+.1f} pts)")
    print("-" * 80)
    print(f" Player {m['player']}'s Hand Before Play ({len(m['hand_before'])} cards):")

    for i, (c_idx, c_name) in enumerate(zip(m["hand_before"], m["hand_names"])):
        played_tag = " <-- PLAYED" if c_idx == m["card_played"] else ""
        print(f"   [{i + 1:2d}] {c_name:<24}{played_tag}")

    print("-" * 80)
    print(" All Player Scores (after this move):")
    for p in range(9):
        role_str = game_data["role_labels"][p]
        score_val = m["scores_after"][p]
        marker = "->" if p == m["player"] else "  "
        print(f"   {marker} Player {p} ({role_str:<15}): {score_val:6.1f} pts")
    print("=" * 80)

def replay_game(game_data, total_games):
    game_data["total_games"] = total_games
    move_idx = 0
    total_moves = len(game_data["moves"])

    while True:
        display_move(game_data, move_idx)
        print("\n Controls: [Enter/N] Next move | [P] Prev move | [J] Jump to turn | [M] Game Menu | [Q] Quit")
        cmd = input(" Command > ").strip().lower()

        if cmd in ["", "n", "next"]:
            if move_idx < total_moves - 1:
                move_idx += 1
            else:
                input("\n [End of Game reached! Press Enter to continue]")
        elif cmd in ["p", "prev", "previous", "b", "back"]:
            if move_idx > 0:
                move_idx -= 1
            else:
                input("\n [Already at Move 1! Press Enter to continue]")
        elif cmd in ["j", "jump"]:
            try:
                target_turn = input(f" Jump to turn number (1-{total_moves}): ").strip()
                val = int(target_turn) - 1
                if 0 <= val < total_moves:
                    move_idx = val
                else:
                    input(f"\n [Invalid turn number! Must be 1 to {total_moves}. Press Enter]")
            except ValueError:
                input("\n [Invalid input! Press Enter]")
        elif cmd in ["m", "menu", "g", "game"]:
            break
        elif cmd in ["q", "quit", "exit"]:
            print("\nExiting Replay Viewer. Goodbye!")
            sys.exit(0)

def main_menu(games_data):
    num_games = len(games_data)
    while True:
        clear_screen()
        print("=" * 80)
        print(" " * 22 + "STAY POSITIVE: GAME REPLAY VIEWER")
        print("=" * 80)
        print(f"{'Game #':<8} | {'Winners':<28} | {'DQN Agent Scores (P0-P8)':<36}")
        print("-" * 80)

        for g in games_data:
            g_num = g["game_num"]
            winners_str = ", ".join([f"P{w} ({g['role_labels'][w]})" for w in g["winners"]])
            dqn_scores = [f"P{p}: {g['final_scores'][p]:.0f}" for p in range(9) if "DQN" in g["role_labels"][p]]
            scores_str = " | ".join(dqn_scores)
            print(f" Game {g_num:<3} | {winners_str:<28} | {scores_str}")

        print("=" * 80)
        print(f" Select a game number [1-{num_games}] to flip through move-by-move, or [Q] to quit.")

        choice = input("\n Select Game > ").strip().lower()
        if choice in ["q", "quit", "exit"]:
            print("\nExiting Replay Viewer. Goodbye!")
            break

        try:
            g_idx = int(choice) - 1
            if 0 <= g_idx < num_games:
                replay_game(games_data[g_idx], num_games)
            else:
                input(f"\n [Invalid selection! Enter a number between 1 and {num_games}. Press Enter]")
        except ValueError:
            input("\n [Invalid input! Enter a valid number or 'q'. Press Enter]")

def main():
    parser = argparse.ArgumentParser(description="Play and Replay 9-Player Stay Positive Games Interactively")
    parser.add_argument("--num-games", type=int, default=10, help="Number of 9-player games to simulate (default: 10)")
    parser.add_argument("--diamond-zero", "--diamond-is-zero", dest="diamond_is_zero", action="store_true",
                        help="Enable rule where effective value of Diamond cards is 0")
    args = parser.parse_args()

    games_data = run_games_in_memory(
        num_games=args.num_games,
        diamond_is_zero=args.diamond_is_zero
    )

    main_menu(games_data)

if __name__ == "__main__":
    main()


