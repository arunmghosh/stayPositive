import random
import math

# Suit Constants
HEARTS = 0
DIAMONDS = 1
SPADES = 2
CLUBS = 3
JOKER = 4

SUIT_NAMES = {
    HEARTS: "Hearts",
    DIAMONDS: "Diamonds",
    SPADES: "Spades",
    CLUBS: "Clubs",
    JOKER: "Joker"
}

VALUE_NAMES = {
    1: "Ace", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10",
    11: "Jack", 12: "Queen", 13: "King", 0: "Joker"
}

def get_card_suit(card_idx):
    if card_idx >= 52:
        return JOKER
    return card_idx // 13

def get_card_value(card_idx):
    if card_idx >= 52:
        return 0  # Joker value is represented as 0
    return (card_idx % 13) + 1

def get_card_effective_value(card_idx, diamond_is_zero=False):
    suit = get_card_suit(card_idx)
    v = get_card_value(card_idx)
    if suit == HEARTS:
        return float(v)
    elif suit == DIAMONDS:
        return 0.0 if diamond_is_zero else float(v)
    elif suit == SPADES:
        return 1.0 / v
    elif suit == CLUBS:
        return float(-v)
    elif suit == JOKER:
        return 0.0

def calculate_turn_score(top_card, card_idx, diamond_is_zero=False):
    if top_card is None:
        if get_card_suit(card_idx) == JOKER:
            return 0.0
        else:
            return float(math.floor(get_card_effective_value(card_idx, diamond_is_zero)))
    else:
        played_suit = get_card_suit(card_idx)
        top_suit = get_card_suit(top_card)
        if played_suit == JOKER or top_suit == JOKER:
            return 0.0
        else:
            T = get_card_effective_value(top_card, diamond_is_zero)
            v = get_card_value(card_idx)
            if played_suit == HEARTS:
                return float(math.floor(T + v))
            elif played_suit == DIAMONDS:
                return float(math.floor(T * v))
            elif played_suit == SPADES:
                return float(math.floor(T / v))
            elif played_suit == CLUBS:
                return float(math.floor(T - v))

def get_card_name(card_idx):
    suit = get_card_suit(card_idx)
    value = get_card_value(card_idx)
    if suit == JOKER:
        return f"Joker_{card_idx - 52}"
    return f"{VALUE_NAMES[value]} of {SUIT_NAMES[suit]}"

class StayPositiveGame:
    def __init__(self, num_players=3, seed=None, diamond_is_zero=False):
        self.num_players = num_players
        self.diamond_is_zero = diamond_is_zero
        if seed is not None:
            random.seed(seed)
        self.reset()

    def reset(self):
        # 54-card deck (0 to 53)
        # 0-51: standard cards, 52-53: Jokers
        deck = list(range(54))
        random.shuffle(deck)

        # Handle deck division
        remainder = len(deck) % self.num_players
        if remainder > 0:
            # Randomly discard enough cards
            self.discarded_cards = deck[:remainder]
            deck = deck[remainder:]
        else:
            self.discarded_cards = []

        # Deal cards
        cards_per_player = len(deck) // self.num_players
        self.hands = [sorted(deck[i * cards_per_player : (i + 1) * cards_per_player]) for i in range(self.num_players)]
        
        # Game state tracking
        self.scores = [0.0] * self.num_players
        self.played_cards = set()
        self.top_card = None  # None indicates empty pile
        
        # Choose starting player randomly
        self.starting_player = random.randint(0, self.num_players - 1)
        self.current_player = self.starting_player
        
        self.turn_count = 0
        self.history = []

    def get_valid_actions(self, player_idx):
        return self.hands[player_idx]

    def play_card(self, card_idx):
        player_idx = self.current_player
        
        # Verify card is in player's hand
        if card_idx not in self.hands[player_idx]:
            raise ValueError(f"Card {get_card_name(card_idx)} is not in Player {player_idx}'s hand.")
        
        # Remove from hand
        self.hands[player_idx].remove(card_idx)
        self.played_cards.add(card_idx)

        # Calculate turn score
        turn_score = calculate_turn_score(self.top_card, card_idx, self.diamond_is_zero)
        
        # Add score
        self.scores[player_idx] += turn_score
        
        # Update pile top card
        prev_top_card = self.top_card
        self.top_card = card_idx
        
        # Record turn in history
        self.history.append({
            'player': player_idx,
            'card': card_idx,
            'card_name': get_card_name(card_idx),
            'prev_top_card': prev_top_card,
            'prev_top_name': get_card_name(prev_top_card) if prev_top_card is not None else None,
            'turn_score': turn_score,
            'total_score': self.scores[player_idx]
        })
        
        # Increment turn count
        self.turn_count += 1
        
        # Determine next player
        if not self.is_game_over():
            self.current_player = (self.current_player + 1) % self.num_players
        
        return turn_score

    def is_game_over(self):
        # All hands empty
        return all(len(hand) == 0 for hand in self.hands)

    def get_winners(self):
        if not self.is_game_over():
            return []
        max_score = max(self.scores)
        # Returns a list of player indices who share the max score (in case of ties)
        return [i for i, score in enumerate(self.scores) if score == max_score]

    def render(self):
        print(f"--- Turn {self.turn_count} ---")
        print(f"Current Player: {self.current_player}")
        print(f"Top Card: {get_card_name(self.top_card) if self.top_card is not None else 'Empty'}")
        print(f"Scores: {self.scores}")
        for i, hand in enumerate(self.hands):
            print(f"Player {i} Hand ({len(hand)} cards): {[get_card_name(c) for c in hand]}")
        print("----------------")
