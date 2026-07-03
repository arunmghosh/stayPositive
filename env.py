import numpy as np
from game import StayPositiveGame, get_card_suit, get_card_value
import math

class StayPositiveEnv:
    def __init__(self, min_players=6, max_players=9, seed=None):
        self.min_players = min_players
        self.max_players = max_players
        self.games = [StayPositiveGame(num_players=i, seed=seed) for i in range(self.min_players, self.max_players + 1)]
        self.current_game_ind = 0  # switch game size every episode
        self.observation_space_size = 165  # 54 + 55 + 54 + 2
        self.action_space_size = 54

    def reset(self):
        active_game = self.games[self.current_game_ind]
        active_game.reset()
        active_player = active_game.current_player
        obs = self._get_observation(active_player)
        mask = self._get_action_mask(active_player)
        done = active_game.is_game_over()
        return obs, active_player, mask, done

    def step(self, action):
        # Get current game
        active_game = self.games[self.current_game_ind]
        
        # Play the card
        active_game.play_card(action)
        
        done = active_game.is_game_over()
        
        rewards = [0.0] * active_game.num_players
        if done:
            winners = active_game.get_winners()
            for p in range(active_game.num_players):
                if p in winners:
                    if len(winners) == 1:
                        rewards[p] = 1.0
                    else:
                        rewards[p] = 0.5  # effective draw case
                else:
                    rewards[p] = -1.0
        
        next_player = active_game.current_player
        next_obs = self._get_observation(next_player)
        next_mask = self._get_action_mask(next_player)
        
        return next_obs, next_player, rewards, done, next_mask

    def _get_observation(self, player_idx):
        active_game = self.games[self.current_game_ind]

        # 1. Player's hand (54-dim)
        hand_vec = np.zeros(54, dtype=np.float32)
        for card in active_game.hands[player_idx]:
            hand_vec[card] = 1.0
            
        # 2. Top card of pile (55-dim)
        top_card_vec = np.zeros(55, dtype=np.float32)
        if active_game.top_card is None:
            top_card_vec[54] = 1.0  # Index 54 represents empty pile
        else:
            top_card_vec[active_game.top_card] = 1.0
            
        # 3. Played cards history (54-dim)
        played_vec = np.zeros(54, dtype=np.float32)
        for card in active_game.played_cards:
            played_vec[card] = 1.0
            
        # 4. Relative scores (max_players - 1 dim)
        # S_j - S_i for all j != i, clockwise order
        rel_score_vec = np.zeros(2, dtype=np.float32)
        relative_scores = []
        num_players = active_game.num_players
        for k in range(1, num_players):
            other_idx = (player_idx + k) % num_players
            score_diff = active_game.scores[other_idx] - active_game.scores[player_idx]
            if k == 1:  # immediate next player
                rel_score_vec[0] = score_diff
            relative_scores.append(score_diff)

        highest_opp_score = max(relative_scores)
        rel_score_vec[1] = highest_opp_score  # winning player 
        
        # Concatenate everything
        obs = np.concatenate([
            hand_vec,
            top_card_vec,
            played_vec,
            rel_score_vec,
        ])
        return obs

    def _get_action_mask(self, player_idx):
        active_game = self.games[self.current_game_ind]
        mask = np.zeros(54, dtype=bool)
        for card in active_game.hands[player_idx]:
            mask[card] = True
        return mask
