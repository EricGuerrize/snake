"""snake_rl — Q-learning em NumPy puro para o jogo Snake."""

from snake_rl.agent import DQNAgent
from snake_rl.env import Direction, SnakeEnv

__version__ = "0.1.0"
__all__ = ["DQNAgent", "Direction", "SnakeEnv", "__version__"]
