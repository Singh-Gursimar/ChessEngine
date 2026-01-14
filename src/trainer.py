"""
Training pipeline for the chess neural network using self-play.
Implements AlphaZero-style training with:
- Self-play game generation
- Replay buffer for experience replay
- Data augmentation (board reflections)
- Temperature-based exploration
- Iterative training loop
- Game logging in PGN format
"""

import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
import chess
import chess.pgn
from collections import deque
import random
import os
from datetime import datetime
import io

from .chess_board import ChessBoard
from .mcts import MCTS
from .neural_network import ChessNeuralNetwork


class ReplayBuffer:
    """Fixed-size replay buffer for storing training examples."""
    
    def __init__(self, max_size: int = 50000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """Add examples to buffer."""
        self.buffer.extend(examples)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch from buffer."""
        batch = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return np.array(states), np.array(policies), np.array(values).reshape(-1, 1)
    
    def get_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get all examples in buffer."""
        if len(self.buffer) == 0:
            return np.array([]), np.array([]), np.array([])
        states, policies, values = zip(*self.buffer)
        return np.array(states), np.array(policies), np.array(values).reshape(-1, 1)
    
    def __len__(self):
        return len(self.buffer)


class GameLogger:
    """Logs games in PGN format for later review."""
    
    def __init__(self, log_dir: str = "games"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a new PGN file for this training session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pgn_path = os.path.join(log_dir, f"training_games_{timestamp}.pgn")
        self.game_count = 0
        
        # Write header
        with open(self.pgn_path, 'w') as f:
            f.write(f"; Chess Engine Training Games\n")
            f.write(f"; Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        print(f"Game log: {self.pgn_path}")
    
    def log_game(self, moves: List[chess.Move], result: str, 
                 iteration: int = 0, game_num: int = 0,
                 move_evals: Optional[List[float]] = None):
        """Log a single game to PGN file.
        
        Args:
            moves: List of moves played.
            result: Game result ('1-0', '0-1', '1/2-1/2', '*').
            iteration: Training iteration number.
            game_num: Game number within iteration.
            move_evals: Optional list of evaluations for each move.
        """
        self.game_count += 1
        
        # Create PGN game
        game = chess.pgn.Game()
        
        # Set headers
        game.headers["Event"] = "Self-Play Training"
        game.headers["Site"] = "Chess Engine"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = f"{iteration}.{game_num}"
        game.headers["White"] = f"Engine_iter{iteration}"
        game.headers["Black"] = f"Engine_iter{iteration}"
        game.headers["Result"] = result if result else "*"
        
        # Add moves
        node = game
        for i, move in enumerate(moves):
            node = node.add_variation(move)
            # Add evaluation as comment if available
            if move_evals and i < len(move_evals):
                node.comment = f"eval: {move_evals[i]:.2f}"
        
        # Append to PGN file
        with open(self.pgn_path, 'a') as f:
            f.write(str(game))
            f.write("\n\n")
    
    def get_log_path(self) -> str:
        """Return path to current log file."""
        return self.pgn_path


class SelfPlayTrainer:
    """Trainer that generates training data through self-play."""
    
    def __init__(self, neural_network: ChessNeuralNetwork,
                 num_simulations: int = 200,
                 c_puct: float = 2.0,
                 replay_buffer_size: int = 50000,
                 use_augmentation: bool = True,
                 log_games: bool = True,
                 log_dir: str = "games"):
        """Initialize the trainer.
        
        Args:
            neural_network: Neural network to train.
            num_simulations: MCTS simulations per move.
            c_puct: Exploration constant for MCTS.
            replay_buffer_size: Maximum size of replay buffer.
            use_augmentation: Whether to use data augmentation.
            log_games: Whether to log games to PGN.
            log_dir: Directory for game logs.
        """
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.mcts = MCTS(neural_network, c_puct, num_simulations)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.use_augmentation = use_augmentation
        
        # Game logging
        self.log_games = log_games
        self.game_logger = GameLogger(log_dir) if log_games else None
        self.current_iteration = 0
        
        # Statistics
        self.games_played = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
    
    def self_play_game(self, temperature: float = 1.0, 
                       temp_threshold: int = 30,
                       game_num: int = 0) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Play a game through self-play.
        
        Args:
            temperature: Temperature for move selection (higher = more exploration).
            temp_threshold: Move number after which temperature drops to near-zero.
            game_num: Game number for logging.
            
        Returns:
            List of (state, policy, value) tuples for training.
        """
        board = ChessBoard()
        history = []
        move_count = 0
        
        # For logging
        game_moves = []
        move_evals = []
        
        while not board.is_game_over():
            move_count += 1
            
            # Use temperature for early moves, then switch to deterministic
            current_temp = temperature if move_count <= temp_threshold else 0.1
            
            # Run MCTS search
            best_move, move_probs, _ = self.mcts.search(board)
            
            # Store state and policy (flip for black's perspective)
            state = board.to_tensor()
            flip = not board.turn  # Flip if black to move
            policy = self._move_probs_to_policy(move_probs, flip)
            history.append((state, policy, board.turn))
            
            # Apply temperature and select move
            if current_temp > 0:
                move = self._select_move_with_temperature(move_probs, current_temp)
            else:
                move = best_move
            
            # Store move and evaluation for logging
            game_moves.append(move)
            # Get evaluation (visit count ratio as proxy for confidence)
            move_evals.append(move_probs.get(move, 0))
            
            board.make_move(move)
            
            # Limit game length to avoid infinite games
            if move_count >= 200:
                break
        
        # Assign values based on game result
        result = board.get_result()
        
        # Handle games that hit move limit (not technically over)
        if result is None:
            result = '1/2-1/2'  # Treat as draw
        
        training_data = []
        
        # Update statistics
        self.games_played += 1
        if result == '1-0':
            self.white_wins += 1
        elif result == '0-1':
            self.black_wins += 1
        else:
            self.draws += 1
        
        # Log game
        if self.game_logger:
            self.game_logger.log_game(
                game_moves, result, 
                self.current_iteration, game_num,
                move_evals
            )
        
        for state, policy, turn in history:
            # turn is True for WHITE, False for BLACK
            if result == '1-0':
                value = 1.0 if turn else -1.0
            elif result == '0-1':
                value = -1.0 if turn else 1.0
            else:
                value = 0.0
            
            training_data.append((state, policy, value))
            
            # Data augmentation: horizontal flip (mirror board)
            if self.use_augmentation:
                flipped_state, flipped_policy = self._augment_horizontal_flip(state, policy)
                training_data.append((flipped_state, flipped_policy, value))
        
        return training_data
    
    def _augment_horizontal_flip(self, state: np.ndarray, 
                                  policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment data by flipping board horizontally.
        
        Chess is symmetric along the a-h axis (files), so we can mirror
        the board to effectively double our training data.
        """
        # Flip state horizontally (reverse columns)
        flipped_state = np.flip(state, axis=1).copy()
        
        # Flip policy: need to mirror from_sq and to_sq
        flipped_policy = np.zeros_like(policy)
        for idx in range(len(policy)):
            if policy[idx] > 0:
                from_sq = idx // 64
                to_sq = idx % 64
                
                # Mirror squares (file = 7 - file)
                from_file = from_sq % 8
                from_rank = from_sq // 8
                to_file = to_sq % 8
                to_rank = to_sq // 8
                
                new_from_sq = from_rank * 8 + (7 - from_file)
                new_to_sq = to_rank * 8 + (7 - to_file)
                
                new_idx = new_from_sq * 64 + new_to_sq
                flipped_policy[new_idx] = policy[idx]
        
        return flipped_state, flipped_policy
    
    def _flip_square(self, square: int) -> int:
        """Flip a square for black's perspective (rotate 180 degrees)."""
        row = square // 8
        col = square % 8
        return (7 - row) * 8 + (7 - col)
    
    def _move_probs_to_policy(self, move_probs: dict, flip: bool = False) -> np.ndarray:
        """Convert move probabilities dict to policy array.
        
        Args:
            move_probs: Dictionary mapping moves to probabilities.
            flip: Whether to flip squares (for black's perspective).
            
        Returns:
            Policy array of shape (4096,).
        """
        policy = np.zeros(ChessNeuralNetwork.POLICY_SIZE, dtype=np.float32)
        
        for move, prob in move_probs.items():
            from_sq = move.from_square
            to_sq = move.to_square
            
            if flip:
                from_sq = self._flip_square(from_sq)
                to_sq = self._flip_square(to_sq)
            
            idx = from_sq * 64 + to_sq
            policy[idx] = prob
        
        return policy
    
    def _select_move_with_temperature(self, move_probs: dict, 
                                       temperature: float) -> chess.Move:
        """Select move using temperature-scaled probabilities.
        
        Args:
            move_probs: Dictionary mapping moves to probabilities.
            temperature: Temperature parameter.
            
        Returns:
            Selected move.
        """
        moves = list(move_probs.keys())
        probs = np.array([move_probs[m] for m in moves])
        
        # Apply temperature
        if temperature != 1.0 and temperature > 0:
            # Use log to avoid numerical issues
            log_probs = np.log(probs + 1e-10)
            scaled_probs = np.exp(log_probs / temperature)
            probs = scaled_probs / scaled_probs.sum()
        
        # Ensure valid probability distribution
        probs = np.clip(probs, 0, 1)
        probs = probs / probs.sum()
        
        idx = np.random.choice(len(moves), p=probs)
        return moves[idx]
    
    def generate_training_data(self, num_games: int,
                               temperature: float = 1.0,
                               temp_threshold: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate training data through self-play.
        
        Args:
            num_games: Number of games to play.
            temperature: Temperature for move selection.
            temp_threshold: Move after which temperature drops.
            
        Returns:
            Tuple of (states, policies, values) arrays.
        """
        all_data = []
        
        for game_num in tqdm(range(num_games), desc="Self-play games"):
            game_data = self.self_play_game(temperature, temp_threshold, game_num + 1)
            all_data.extend(game_data)
            
            # Print periodic statistics
            if (game_num + 1) % 10 == 0:
                win_rate = (self.white_wins + self.black_wins) / max(1, self.games_played)
                draw_rate = self.draws / max(1, self.games_played)
                tqdm.write(f"  Games: {self.games_played}, "
                          f"W: {self.white_wins}, B: {self.black_wins}, D: {self.draws}, "
                          f"Decisive: {win_rate:.1%}, Draws: {draw_rate:.1%}")
        
        # Add to replay buffer
        self.replay_buffer.add(all_data)
        
        # Return all data from buffer for training
        return self.replay_buffer.get_all()
    
    def train_iteration(self, num_games: int = 100, epochs: int = 5,
                        batch_size: int = 64, temperature: float = 1.0,
                        temp_threshold: int = 30) -> dict:
        """Run one training iteration (self-play + training).
        
        Args:
            num_games: Number of self-play games.
            epochs: Training epochs.
            batch_size: Training batch size.
            temperature: Move selection temperature.
            temp_threshold: Move after which temperature drops.
            
        Returns:
            Training history dictionary.
        """
        self.current_iteration += 1
        
        print(f"\n--- Generating {num_games} self-play games ---")
        states, policies, values = self.generate_training_data(
            num_games, temperature, temp_threshold
        )
        
        print(f"\n--- Training on {len(states)} positions ---")
        print(f"    (Replay buffer size: {len(self.replay_buffer)})")
        
        if len(states) == 0:
            print("No training data generated!")
            return {}
        
        history = self.neural_network.train(states, policies, values, epochs, batch_size)
        
        # Print final game statistics
        print(f"\n--- Game Statistics ---")
        print(f"Total games: {self.games_played}")
        print(f"White wins: {self.white_wins} ({100*self.white_wins/max(1,self.games_played):.1f}%)")
        print(f"Black wins: {self.black_wins} ({100*self.black_wins/max(1,self.games_played):.1f}%)")
        print(f"Draws: {self.draws} ({100*self.draws/max(1,self.games_played):.1f}%)")
        
        if self.game_logger:
            print(f"Games logged to: {self.game_logger.get_log_path()}")
        
        return history
    
    def train(self, num_games: int = 100, epochs: int = 10,
              batch_size: int = 32, temperature: float = 1.0) -> None:
        """Run training loop (legacy interface).
        
        Args:
            num_games: Number of self-play games.
            epochs: Training epochs per iteration.
            batch_size: Training batch size.
            temperature: Move selection temperature.
        """
        self.train_iteration(num_games, epochs, batch_size, temperature)
