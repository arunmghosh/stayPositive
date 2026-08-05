# Stay Positive
I trained a Deep Q Network (DQN) to play a card game I invented call "Stay Positive", in order to gauge the level of strategy v.s. luck
involved in the game, and to search for evidence that a reinforcement learning (RL) model can discover a strategy when none exist yet. This
experiment showed that AI models can learn to strategize and think ahead even in games with a component of luck that could introduce noise 
in a learning environment. 

[See "Stay Positive Project" Powerpoint for more information]

## Overview
AI models have already discovered new winning strategies in games such as Chess and Go, since they have a much greater capacity for
analyzing and memorizing positions, and an incredibly faster processing time. But why is there still value in humans playing games, even 
when we can program AI engines to win for us? I believe that games are a way to express creativity, and creativity now lies in the ways we
think to apply AI algorithms. I developed a brand new card game, "Stay Positive", with little to no knowledge of how much strategy was 
involved in playing it. Instead of simply training a DQN to win every time, I wanted to see if it could identify the level of strategy v.s.
luck a player might need to win. My metric for gauging the level of strategy was evidence of deliberate behavior in the DQN's choices over
time during training. This project demonstrated an evolution from random choices to simple greedy decisions, and then to forward-thinking
decisions which demonstrated a clear strategy component to the game. 

## Results
I tested two variations of the game (3-player, and 6-9 player), since fewer players gave each individual more cards and more opportunity for
strategy (see Powerpoint for rules). Furthermore, I tested the DQN against random and greedy (moved to gain the most points immediately, 
sometimes sacrificing long term gain) baselines, as well as myself. 

In the 3-player version, the DQN win rate evolved from 7.5% to 69.5% over 12,000 training episodes, while Greedy fell from 81% to 23%, and
random fell from 11.5% to 8%. Such a dominant performance is clear evidence that strategy exists in "Stay Positive". 

In the 6-9 player version, where the players had fewer cards and fewer chances to act strategically, this trend collapsed. There was a clear
element of luck which made the win rate an unreliable statistic for evidence of learning. Instead, I looked at the average raw score in a 
game. While the DQN significantly outperformed random bots in the many-player games, greedy agents still scored higher (on average), due to 
inadvertent collusion (greedy is a best response to greedy). Nonetheless, the positive difference from random still suggests deliberate 
choices made to improve the score. 

## Approach

### 1. Data Formulation
The input to the model was the state of the game each time the DQN agent had to make a decision (play a card). This consisted of the player hand, top card, cards already played, relative score v.s. leader, and relative score v.s. next player. The output was a single decision: which card in the player hand should be played.  

### 2. Model Architecture
I chose to use a Deep Q Network (DQN) because it would not be practical to know the entire Q-table for "Stay Positive" given the number of permutations in which the 54 cards could be played. There was also no game history that a traditional feed-forward neural network could learn from, since the game had just been invented, so a reinforcement-learning model was ideal.  

### 3. Training & Evaluation
I trained the model by having it play against greedy and random bots in a total of 12000 episodes. The performance against random was meant to gauge if there was a strategy component at all, and performance against greedy (a strategy I came up with) was meant to measure the efficacy of the DQN's strategy. 

I used a policy and target DQN structure with a replay buffer capacity of 100000, fixed learning rate of 0.0001, batch size of 64, tau = 0.005 (soft-update rate), and gamma = 0.99 (future reward multiplier, so model could think ahead). The performance of the model was initially evaluated by the win rate for the 3-player version, but this metric collapsed in the 6-9 player version. Instead, I looked at the average score and consistency of score, in which the DQN surpassed random but not greedy. 

## What "Stay Positive" Looks Like
Show an example input → model output.

## Installation
```bash
git clone ...
cd ...
pip install -r requirements.txt
