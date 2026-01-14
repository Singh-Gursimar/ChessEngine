"""
Main Chess Engine interface combining all components.
"""

import chess
from typing import Optional, Tuple

from .chess_board import ChessBoard
from .mcts import MCTS
from .neural_network import ChessNeuralNetwork


class ChessEngine:
    """Main chess engine class."""
    
    def __init__(self, model_path: Optional[str] = None,
                 num_simulations: int = 800):
        """Initialize the chess engine.
        
        Args:
            model_path: Optional path to load pre-trained model.
            num_simulations: Number of MCTS simulations per move.
        """
        self.neural_network = ChessNeuralNetwork(model_path)
        self.mcts = MCTS(self.neural_network, num_simulations=num_simulations)
        self.board = ChessBoard()
    
    def new_game(self, fen: Optional[str] = None) -> None:
        """Start a new game.
        
        Args:
            fen: Optional FEN string for starting position.
        """
        self.board = ChessBoard(fen)
    
    def get_best_move(self) -> Tuple[chess.Move, float]:
        """Get the best move for the current position.
        
        Returns:
            Tuple of (best move, win probability estimate from MCTS).
        """
        best_move, move_probs, win_prob = self.mcts.search(self.board)
        return best_move, win_prob
    
    def make_move(self, move: chess.Move) -> bool:
        """Make a move on the board.
        
        Args:
            move: The move to make.
            
        Returns:
            True if move was legal.
        """
        return self.board.make_move(move)
    
    def make_move_uci(self, uci_string: str) -> bool:
        """Make a move using UCI notation.
        
        Args:
            uci_string: Move in UCI format (e.g., 'e2e4') or castling notation (O-O, 0-0).
            
        Returns:
            True if move was legal.
        """
        try:
            # Handle castling notation (O-O, 0-0, o-o for kingside; O-O-O, 0-0-0, o-o-o for queenside)
            uci_lower = uci_string.lower().replace('0', 'o')
            
            if uci_lower in ['o-o', 'oo']:
                # Kingside castling
                if self.board.turn:  # White
                    move = chess.Move.from_uci('e1g1')
                else:  # Black
                    move = chess.Move.from_uci('e8g8')
            elif uci_lower in ['o-o-o', 'ooo']:
                # Queenside castling
                if self.board.turn:  # White
                    move = chess.Move.from_uci('e1c1')
                else:  # Black
                    move = chess.Move.from_uci('e8c8')
            else:
                move = chess.Move.from_uci(uci_string)
            
            return self.make_move(move)
        except ValueError:
            return False
    
    def get_legal_moves(self) -> list:
        """Get all legal moves in current position."""
        return self.board.get_legal_moves()
    
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.board.is_game_over()
    
    def get_result(self) -> Optional[str]:
        """Get game result."""
        return self.board.get_result()
    
    def get_fen(self) -> str:
        """Get FEN of current position."""
        return self.board.fen
    
    def display(self) -> str:
        """Get string representation of the board."""
        return str(self.board)
    
    def save_model(self, path: str) -> None:
        """Save the neural network model.
        
        Args:
            path: File path to save model.
        """
        self.neural_network.save(path)
