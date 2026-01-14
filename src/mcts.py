"""
Monte Carlo Tree Search (MCTS) implementation for chess.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import chess

from .chess_board import ChessBoard
from .piece_values import evaluate_position, is_endgame


class MCTSNode:
    """A node in the MCTS tree."""
    
    def __init__(self, board: ChessBoard, parent: Optional['MCTSNode'] = None,
                 move: Optional[chess.Move] = None, prior: float = 0.0):
        """Initialize MCTS node.
        
        Args:
            board: The chess board state at this node.
            parent: Parent node (None for root).
            move: The move that led to this state.
            prior: Prior probability from neural network policy.
        """
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
    
    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float = 1.41) -> float:
        """Calculate Upper Confidence Bound score.
        
        Args:
            c_puct: Exploration constant.
            
        Returns:
            UCB score for node selection.
        """
        if self.parent is None:
            return 0.0
        
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        # Negate value since child's value is from opponent's perspective
        # A position bad for opponent (negative value) is good for us
        return -self.value + exploration
    
    def select_child(self, c_puct: float = 1.41) -> 'MCTSNode':
        """Select child with highest UCB score.
        
        Args:
            c_puct: Exploration constant.
            
        Returns:
            Child node with highest UCB score.
        """
        return max(self.children.values(), key=lambda n: n.ucb_score(c_puct))
    
    def expand(self, policy: Dict[chess.Move, float], board_obj: 'ChessBoard' = None) -> None:
        """Expand node by creating children for all legal moves.
        
        Uses strong chess heuristics to guide move ordering, especially important
        when the neural network is untrained.
        
        Args:
            policy: Dictionary mapping moves to prior probabilities.
            board_obj: Optional ChessBoard for move ordering bonuses.
        """
        board = self.board.board
        is_white = board.turn
        move_count = len(board.move_stack)
        is_opening = move_count < 20
        is_endgame = self._is_endgame(board)
        
        # Piece values for calculations
        PIECE_VALUES = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
        }
        
        for move in self.board.get_legal_moves():
            if move not in self.children:
                child_board = self.board.copy()
                child_board.make_move(move)
                prior = policy.get(move, 0.01)  # Small default prior
                
                piece = board.piece_at(move.from_square)
                piece_type = piece.piece_type if piece else None
                from_sq = move.from_square
                to_sq = move.to_square
                
                # =============================================================
                # 1. CHECKMATE - Absolute priority
                # =============================================================
                if child_board.board.is_checkmate():
                    prior = 10000.0
                    self.children[move] = MCTSNode(child_board, self, move, prior)
                    continue
                
                # =============================================================
                # 2. CAPTURES - MVV-LVA with SEE approximation
                # =============================================================
                if board.is_capture(move):
                    captured = board.piece_at(to_sq)
                    # Handle en passant
                    if captured is None and piece_type == chess.PAWN:
                        captured_val = PIECE_VALUES[chess.PAWN]
                    else:
                        captured_val = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
                    
                    attacker_val = PIECE_VALUES.get(piece_type, 0)
                    
                    # Check if target square is defended
                    is_defended = bool(board.attackers(not board.turn, to_sq))
                    
                    if not is_defended:
                        # FREE PIECE - very high priority
                        prior = 50.0 + captured_val / 50.0
                    elif captured_val > attacker_val:
                        # Winning capture
                        prior = 30.0 + (captured_val - attacker_val) / 50.0
                    elif captured_val == attacker_val:
                        # Equal trade - often good
                        prior = 15.0
                    else:
                        # Losing capture - low priority but not zero
                        prior = 2.0
                
                # =============================================================
                # 3. CHECKS - Strong boost (but avoid pointless checks)
                # =============================================================
                elif board.gives_check(move):
                    # Check if it's a useful check (not easily blocked)
                    prior = max(prior, 8.0)
                    # Discovered check or double check is very strong
                    if piece_type != chess.QUEEN:  # Queen checks are often just trades
                        prior *= 1.5
                
                # =============================================================
                # 4. OPENING PRINCIPLES (first 20 moves)
                # =============================================================
                elif is_opening:
                    # --- CASTLING: Very important ---
                    if board.is_castling(move):
                        prior = 25.0  # Very high priority
                    
                    # --- CENTRAL PAWNS: e4, d4, e5, d5 ---
                    elif piece_type == chess.PAWN:
                        to_file = chess.square_file(to_sq)
                        to_rank = chess.square_rank(to_sq)
                        
                        # e4, d4 for white; e5, d5 for black
                        if to_sq in [chess.E4, chess.D4, chess.E5, chess.D5]:
                            prior = max(prior, 12.0)
                        elif to_sq in [chess.E3, chess.D3, chess.E6, chess.D6]:
                            prior = max(prior, 6.0)
                        # c4, c5 (English/Sicilian) are also good
                        elif to_sq in [chess.C4, chess.C5]:
                            prior = max(prior, 8.0)
                        # Avoid moving flank pawns early (a, h pawns)
                        elif to_file in [0, 7] and to_rank < 4:
                            prior = max(prior * 0.3, 0.5)
                    
                    # --- KNIGHT DEVELOPMENT ---
                    elif piece_type == chess.KNIGHT:
                        # Best squares: c3, f3, c6, f6
                        if to_sq in [chess.C3, chess.F3, chess.C6, chess.F6]:
                            prior = max(prior, 10.0)
                        # Nc3 or Nf3 from starting position
                        elif to_sq in [chess.D2, chess.E2]:  # Bad knight squares
                            prior = max(prior * 0.5, 1.0)
                        # Knights on the rim are dim
                        elif chess.square_file(to_sq) in [0, 7]:
                            prior = max(prior * 0.4, 0.5)
                    
                    # --- BISHOP DEVELOPMENT ---
                    elif piece_type == chess.BISHOP:
                        to_rank = chess.square_rank(to_sq)
                        # Fianchetto (b2, g2, b7, g7)
                        if to_sq in [chess.B2, chess.G2, chess.B7, chess.G7]:
                            prior = max(prior, 8.0)
                        # Active diagonals (c4, f4, c5, f5, b5, g5, etc)
                        elif to_rank in [3, 4] and chess.square_file(to_sq) in [1, 2, 5, 6]:
                            prior = max(prior, 7.0)
                        # Don't move bishop multiple times in opening
                        elif len([m for m in board.move_stack if board.piece_at(m.to_square) and 
                                  board.piece_at(m.to_square).piece_type == chess.BISHOP]) > 1:
                            prior = max(prior * 0.7, 1.0)
                    
                    # --- QUEEN: Don't develop too early ---
                    elif piece_type == chess.QUEEN:
                        if move_count < 10:
                            prior = max(prior * 0.2, 0.5)  # Heavily penalize early queen moves
                        else:
                            prior = max(prior * 0.5, 1.0)
                    
                    # --- ROOK: Usually shouldn't move in opening ---
                    elif piece_type == chess.ROOK:
                        # Unless connecting rooks after castling
                        if move_count < 12:
                            prior = max(prior * 0.3, 0.5)
                    
                    # --- KING: Don't move unless castling ---
                    elif piece_type == chess.KING:
                        if not board.is_castling(move):
                            prior = max(prior * 0.1, 0.2)  # Very bad to move king
                
                # =============================================================
                # 5. MIDDLEGAME PRINCIPLES
                # =============================================================
                elif not is_endgame:
                    # Castling still good if available
                    if board.is_castling(move):
                        prior = max(prior, 15.0)
                    
                    # Rooks on open files
                    elif piece_type == chess.ROOK:
                        to_file = chess.square_file(to_sq)
                        file_pawns = 0
                        for rank in range(8):
                            p = board.piece_at(chess.square(to_file, rank))
                            if p and p.piece_type == chess.PAWN:
                                file_pawns += 1
                        if file_pawns == 0:
                            prior = max(prior, 6.0)  # Open file
                        elif file_pawns == 1:
                            prior = max(prior, 4.0)  # Semi-open
                    
                    # Rook on 7th rank
                    elif piece_type == chess.ROOK:
                        to_rank = chess.square_rank(to_sq)
                        if (is_white and to_rank == 6) or (not is_white and to_rank == 1):
                            prior = max(prior, 7.0)
                    
                    # Central control
                    if to_sq in [chess.D4, chess.E4, chess.D5, chess.E5]:
                        prior = max(prior, prior * 1.5)
                
                # =============================================================
                # 6. ENDGAME PRINCIPLES
                # =============================================================
                elif is_endgame:
                    # King should be active in endgame
                    if piece_type == chess.KING:
                        # Move towards center
                        to_file = chess.square_file(to_sq)
                        to_rank = chess.square_rank(to_sq)
                        center_dist = abs(3.5 - to_file) + abs(3.5 - to_rank)
                        prior = max(prior, 5.0 - center_dist * 0.5)
                    
                    # Passed pawn advancement is critical
                    elif piece_type == chess.PAWN:
                        to_rank = chess.square_rank(to_sq)
                        if is_white and to_rank >= 5:
                            prior = max(prior, 8.0 + (to_rank - 4) * 3)
                        elif not is_white and to_rank <= 2:
                            prior = max(prior, 8.0 + (3 - to_rank) * 3)
                
                # =============================================================
                # 7. PENALTIES FOR BAD MOVES
                # =============================================================
                
                # Don't block check with valuable pieces if pawn can do it
                if board.is_check():
                    # We must get out of check - prioritize by piece value
                    if piece_type == chess.QUEEN:
                        # Check if there's a cheaper way to block/escape
                        prior = max(prior * 0.3, 1.0)
                    elif piece_type == chess.ROOK:
                        prior = max(prior * 0.5, 1.0)
                    # Moving king is often necessary
                    elif piece_type == chess.KING:
                        prior = max(prior, 5.0)
                
                # Penalize moving same piece twice in opening
                if is_opening and len(board.move_stack) >= 2:
                    last_move = board.move_stack[-2] if len(board.move_stack) >= 2 else None
                    if last_move and last_move.to_square == from_sq:
                        prior = max(prior * 0.4, 0.5)
                
                # Penalize putting pieces on attacked squares
                if board.is_attacked_by(not board.turn, to_sq):
                    attackers = board.attackers(not board.turn, to_sq)
                    defenders = board.attackers(board.turn, to_sq)
                    if len(list(attackers)) > len(list(defenders)):
                        piece_val = PIECE_VALUES.get(piece_type, 0)
                        if piece_val > 100:  # Not just a pawn
                            prior = max(prior * 0.3, 0.5)
                
                # Ensure minimum prior
                prior = max(prior, 0.01)
                
                self.children[move] = MCTSNode(child_board, self, move, prior)
        
        self.is_expanded = True
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Check if position is an endgame."""
        # Count major/minor pieces (excluding pawns and kings)
        piece_count = 0
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            piece_count += len(board.pieces(piece_type, chess.WHITE))
            piece_count += len(board.pieces(piece_type, chess.BLACK))
        return piece_count <= 6  # Few pieces left
    
    def backpropagate(self, value: float) -> None:
        """Backpropagate value up the tree.
        
        Args:
            value: The value to propagate (from perspective of current player).
        """
        node = self
        while node is not None:
            node.visit_count += 1
            # Flip value for opponent's perspective
            node.value_sum += value
            value = -value
            node = node.parent


class MCTS:
    """Monte Carlo Tree Search algorithm."""
    
    def __init__(self, neural_network, c_puct: float = 2.0, 
                 num_simulations: int = 800):
        """Initialize MCTS.
        
        Args:
            neural_network: Neural network for policy and value prediction.
            c_puct: Exploration constant for UCB.
            num_simulations: Number of simulations per search.
        """
        self.neural_network = neural_network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
    
    def search(self, board: ChessBoard) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        """Run MCTS search from given position.
        
        Args:
            board: Current board state.
            
        Returns:
            Tuple of (best move, move probabilities).
        """
        root = MCTSNode(board.copy())
        
        # Expand root with move ordering
        policy, _ = self._evaluate(root.board)
        root.expand(policy, root.board)
        
        for _ in range(self.num_simulations):
            node = root
            
            # Selection - traverse tree to leaf
            while node.is_expanded and not node.board.is_game_over():
                node = node.select_child(self.c_puct)
            
            # Evaluate leaf
            if node.board.is_game_over():
                value = self._get_terminal_value(node.board)
            else:
                policy, value = self._evaluate(node.board)
                node.expand(policy, node.board)
            
            # Backpropagation
            node.backpropagate(value)
        
        # Calculate move probabilities from visit counts
        move_probs = {}
        total_visits = sum(child.visit_count for child in root.children.values())
        for move, child in root.children.items():
            move_probs[move] = child.visit_count / total_visits if total_visits > 0 else 0
        
        # Select best move by visit count
        best_child = max(root.children.values(), key=lambda x: x.visit_count)
        best_move = best_child.move
        
        # Calculate win probability from best child's value
        # The child's value is from opponent's perspective, so negate it
        win_prob = (-best_child.value + 1) / 2
        
        return best_move, move_probs, win_prob
    
    def _evaluate(self, board: ChessBoard) -> Tuple[Dict[chess.Move, float], float]:
        """Evaluate position using neural network combined with heuristic.
        
        Args:
            board: Board state to evaluate.
            
        Returns:
            Tuple of (policy dict, value).
        """
        legal_moves = board.get_legal_moves()
        
        # Always compute heuristic evaluation (it's reliable)
        heuristic_value = self._evaluate_position_heuristic(board)
        
        if self.neural_network is None:
            # Use uniform policy if no network
            policy = {move: 1.0 / len(legal_moves) for move in legal_moves}
            return policy, heuristic_value
        
        # Get neural network prediction
        state_tensor = board.to_tensor()
        policy_logits, nn_value = self.neural_network.predict(state_tensor)
        
        # Convert policy logits to move probabilities
        # Need to flip move indices when black to move (board is flipped)
        flip = not board.turn
        policy = {}
        for move in legal_moves:
            move_idx = self._move_to_index(move, flip)
            policy[move] = policy_logits[move_idx]
        
        # Normalize
        total = sum(policy.values())
        if total > 0:
            policy = {m: p / total for m, p in policy.items()}
        
        # Blend neural network value with heuristic
        # Use 95% heuristic when NN is untrained/weak - the heuristics are reliable
        # As NN gets trained, you can adjust this ratio
        blended_value = 0.95 * heuristic_value + 0.05 * nn_value
        
        return policy, blended_value
    
    def _get_terminal_value(self, board: ChessBoard) -> float:
        """Get value for terminal position.
        
        Args:
            board: Terminal board state.
            
        Returns:
            1.0 for win, -1.0 for loss, 0.0 for draw.
        """
        result = board.get_result()
        # board.turn is True for WHITE, False for BLACK
        if result == '1-0':
            return 1.0 if board.turn else -1.0
        elif result == '0-1':
            return -1.0 if board.turn else 1.0
        return 0.0
    
    def _evaluate_position_heuristic(self, board: ChessBoard) -> float:
        """Heuristically evaluate a position using advanced evaluation.
        
        Args:
            board: Board position to evaluate.
            
        Returns:
            Value from -1.0 to 1.0 representing position evaluation
            from the perspective of the player to move.
        """
        evaluation = evaluate_position(board.board)
        
        # evaluate_position returns value from WHITE's perspective
        # Convert to current player's perspective
        if not board.board.turn:  # Black to move
            evaluation = -evaluation
        
        # Normalize evaluation to roughly -1.0 to 1.0
        # Scale factor: 400 centipawns (4 pawns) maps to ~1.0
        # Using tanh-like scaling for smoother boundaries
        normalized = evaluation / 400.0
        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, normalized))
    
    def _flip_square(self, square: int) -> int:
        """Flip a square for black's perspective (rotate 180 degrees)."""
        row = square // 8
        col = square % 8
        return (7 - row) * 8 + (7 - col)
    
    def _move_to_index(self, move: chess.Move, flip: bool = False) -> int:
        """Convert move to policy index.
        
        Args:
            move: Chess move.
            flip: Whether to flip squares (for black's perspective).
            
        Returns:
            Index in policy vector.
        """
        from_sq = move.from_square
        to_sq = move.to_square
        
        if flip:
            from_sq = self._flip_square(from_sq)
            to_sq = self._flip_square(to_sq)
        
        # Simple encoding: from_square * 64 + to_square
        return from_sq * 64 + to_sq
