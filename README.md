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

### 1. Data / Problem Formulation
Explain the input and output.

### 2. Model Architecture
Explain the models and why you chose them.

### 3. Training / Evaluation
Explain how you trained and evaluated the system.

## Experiments
Describe the experiments you ran and what you learned.

## Results
Include tables, graphs, and visualizations.

## Example
Show an example input → model output.

## Installation
```bash
git clone ...
cd ...
pip install -r requirements.txt
