# RL Chopsticks
RL Chopsticks (RLC) is a reinforcement learning agent trained to play the game [Chopsticks](https://en.wikipedia.org/wiki/Chopsticks_(hand_game)) with the default rule set. To play against it, run in your terminal: ```python3 game.py```. More specifically:
- ```python3 game.py```: Play against RLC
- ```python3 game.py self-play```: Train RLC via self-play
- ```python3 game.py minimax```: Let RLC play against a Minimax agent (experimental)

## Implementation
RLC was trained using using TD-Learning and self-play. To balance exploration and exploitation, I used UCB1 as it was convenient and effective. I also used ```pypy``` to speed up the code. The agents seem to converge to an optimal policy within 3000 episodes.

## Limitations
RLC is incapable of beating Minimax. The goal is for it to defeat Minimax with an 8-step lookahead. My current plan is to implement an Monte Carlo Tree Search agent to defeat the Minimax agent.