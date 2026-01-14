"""
Chess Board representation and game state management.
Uses python-chess library for move generation and validation.
"""

import chess
import numpy as np
from typing import List, Tuple, Optional

from .piece_values import evaluate_material, evaluate_position, is_endgame


class ChessBoard:
    """Wrapper around python-chess board with additional utilities for ML."""
    
    def __init__(self, fen: Optional[str] = None):
        """Initialize the chess board.
        
        Args:
            fen: Optional FEN string to initialize board state.
        """
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
    
    def get_legal_moves(self) -> List[chess.Move]:
        """Get all legal moves for the current position."""
        return list(self.board.legal_moves)
    
    def make_move(self, move: chess.Move) -> bool:
        """Make a move on the board.
        
        Args:
            move: The move to make.
            
        Returns:
            True if move was legal and made, False otherwise.
        """
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False
    
    def undo_move(self) -> Optional[chess.Move]:
        """Undo the last move.
        
        Returns:
            The move that was undone, or None if no moves to undo.
        """
        if self.board.move_stack:
            return self.board.pop()
        return None
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.board.is_game_over()
    
    def get_result(self) -> Optional[str]:
        """Get the game result.
        
        Returns:
            '1-0' for white win, '0-1' for black win, '1/2-1/2' for draw,
            None if game is not over.
        """
        if not self.is_game_over():
            return None
        return self.board.result()
    
    def to_tensor(self) -> np.ndarray:
        """Convert board state to tensor representation for neural network.
        
        The board is always represented from the perspective of the player to move.
        This means when black is to move, the board is flipped so that black's
        pieces appear on the bottom (ranks 1-2) just like white's pieces when
        white is to move. This allows the network to learn symmetric patterns.
        
        Returns:
            numpy array of shape (8, 8, 14) representing:
            - 6 channels for current player's pieces (P, N, B, R, Q, K)
            - 6 channels for opponent's pieces (p, n, b, r, q, k)
            - 1 channel for current player indicator (all 1s)
            - 1 channel for castling rights
        """
        tensor = np.zeros((8, 8, 14), dtype=np.float32)
        
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # Determine if we need to flip (when black to move)
        flip = not self.board.turn  # True if black to move
        current_player = self.board.turn
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                row = square // 8
                col = square % 8
                
                # Flip board for black's perspective
                if flip:
                    row = 7 - row
                    col = 7 - col
                
                channel = piece_map[piece.piece_type]
                
                # Channels 0-5: current player's pieces
                # Channels 6-11: opponent's pieces
                if piece.color != current_player:
                    channel += 6
                    
                tensor[row, col, channel] = 1.0
        
        # Current player channel (always 1 since we represent from current player's view)
        tensor[:, :, 12] = 1.0
        
        # Castling rights channel (from current player's perspective)
        if flip:
            # Black to move - show black's castling on bottom
            if self.board.has_kingside_castling_rights(chess.BLACK):
                tensor[0, 0, 13] = 1.0  # h1 equivalent for black
            if self.board.has_queenside_castling_rights(chess.BLACK):
                tensor[0, 7, 13] = 1.0  # a1 equivalent for black
            if self.board.has_kingside_castling_rights(chess.WHITE):
                tensor[7, 0, 13] = 1.0  # h8 equivalent for white
            if self.board.has_queenside_castling_rights(chess.WHITE):
                tensor[7, 7, 13] = 1.0  # a8 equivalent for white
        else:
            # White to move - normal orientation
            if self.board.has_kingside_castling_rights(chess.WHITE):
                tensor[0, 7, 13] = 1.0
            if self.board.has_queenside_castling_rights(chess.WHITE):
                tensor[0, 0, 13] = 1.0
            if self.board.has_kingside_castling_rights(chess.BLACK):
                tensor[7, 7, 13] = 1.0
            if self.board.has_queenside_castling_rights(chess.BLACK):
                tensor[7, 0, 13] = 1.0
        
        return tensor
    
    def copy(self) -> 'ChessBoard':
        """Create a copy of the current board state."""
        new_board = ChessBoard()
        new_board.board = self.board.copy()
        return new_board
    
    def __str__(self) -> str:
        """String representation of the board."""
        return str(self.board)
    
    @property
    def fen(self) -> str:
        """Get the FEN string of current position."""
        return self.board.fen()
    
    @property
    def turn(self) -> bool:
        """Get current player (True = White, False = Black)."""
        return self.board.turn
    
    def evaluate_material(self) -> int:
        """Get material balance in centipawns (positive = white advantage)."""
        return evaluate_material(self.board)
    
    def evaluate_position(self) -> int:
        """Get position evaluation in centipawns including piece placement."""
        endgame = is_endgame(self.board)
        return evaluate_position(self.board, endgame)
    
    def is_endgame(self) -> bool:
        """Check if position is in endgame phase."""
        return is_endgame(self.board)
