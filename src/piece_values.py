"""
Advanced Chess Evaluation Function
Implements a comprehensive heuristic evaluation with:
- Tapered evaluation (smooth middlegame to endgame transition)
- Advanced piece-square tables
- Pawn structure analysis (doubled, isolated, passed, backward pawns)
- King safety with attack zones
- Piece coordination and mobility
- Rook/Queen on open files and 7th rank
- Knight outposts
- Bishop pair and bad bishop detection
- Space advantage
- Tempo and development
"""

import chess
from typing import Dict, List, Tuple

# =============================================================================
# PIECE VALUES (in centipawns)
# Using well-tested values from top engines
# =============================================================================

# Middlegame piece values
MG_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 950,
    chess.KING: 20000,
}

# Endgame piece values (knights weaker, rooks stronger)
EG_PIECE_VALUES = {
    chess.PAWN: 110,
    chess.KNIGHT: 280,
    chess.BISHOP: 340,
    chess.ROOK: 550,
    chess.QUEEN: 1000,
    chess.KING: 20000,
}

# For compatibility
PIECE_VALUES = MG_PIECE_VALUES

# =============================================================================
# PIECE-SQUARE TABLES (Middlegame)
# From White's perspective, a1=0, h8=63
# =============================================================================

MG_PAWN_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
     -6,  -4,   1, -24, -24,   1,  -4,  -6,
     -4,  -4,   1,   5,   5,   1,  -4,  -4,
     -6,  -4,   5,  10,  10,   5,  -4,  -6,
     -6,  -4,   2,   8,   8,   2,  -4,  -6,
     -6,  -4,   1,   2,   2,   1,  -4,  -6,
     -6,  -4,   1,   1,   1,   1,  -4,  -6,
      0,   0,   0,   0,   0,   0,   0,   0,
]

MG_KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

MG_BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

MG_ROOK_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
]

MG_QUEEN_TABLE = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

MG_KING_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

# =============================================================================
# PIECE-SQUARE TABLES (Endgame)
# =============================================================================

EG_PAWN_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 134, 158, 173, 178,
     94, 100,  85,  67,  67,  85, 100,  94,
     32,  24,  13,   5,   5,  13,  24,  32,
     13,   9,  -3,  -7,  -7,  -3,   9,  13,
      4,   7,  -6,   1,   1,  -6,   7,   4,
      5,  -4,  -6,  -8,  -8,  -6,  -4,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]

EG_KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

EG_BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

EG_ROOK_TABLE = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
]

EG_QUEEN_TABLE = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]

EG_KING_TABLE = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]

# Combine into dicts for easy access
MG_TABLES = {
    chess.PAWN: MG_PAWN_TABLE,
    chess.KNIGHT: MG_KNIGHT_TABLE,
    chess.BISHOP: MG_BISHOP_TABLE,
    chess.ROOK: MG_ROOK_TABLE,
    chess.QUEEN: MG_QUEEN_TABLE,
    chess.KING: MG_KING_TABLE,
}

EG_TABLES = {
    chess.PAWN: EG_PAWN_TABLE,
    chess.KNIGHT: EG_KNIGHT_TABLE,
    chess.BISHOP: EG_BISHOP_TABLE,
    chess.ROOK: EG_ROOK_TABLE,
    chess.QUEEN: EG_QUEEN_TABLE,
    chess.KING: EG_KING_TABLE,
}

# For compatibility
PIECE_SQUARE_TABLES = MG_TABLES
KING_MIDDLEGAME_TABLE = MG_KING_TABLE
KING_ENDGAME_TABLE = EG_KING_TABLE
PAWN_TABLE = MG_PAWN_TABLE
KNIGHT_TABLE = MG_KNIGHT_TABLE
BISHOP_TABLE = MG_BISHOP_TABLE
ROOK_TABLE = MG_ROOK_TABLE
QUEEN_TABLE = MG_QUEEN_TABLE

# =============================================================================
# EVALUATION CONSTANTS
# =============================================================================

# Pawn structure
DOUBLED_PAWN_PENALTY = -15
ISOLATED_PAWN_PENALTY = -20
BACKWARD_PAWN_PENALTY = -10
PASSED_PAWN_BONUS = [0, 10, 20, 40, 60, 90, 130, 0]  # By rank (0-7)
CONNECTED_PASSED_BONUS = 20
PROTECTED_PASSED_BONUS = 15

# King safety
PAWN_SHIELD_BONUS = 10
PAWN_STORM_PENALTY = -5
KING_OPEN_FILE_PENALTY = -25
KING_SEMI_OPEN_FILE_PENALTY = -15
KING_ATTACK_WEIGHT = [0, 0, 50, 75, 88, 94, 97, 99, 99, 99, 99, 99, 99, 99, 99, 99]

# Piece bonuses
BISHOP_PAIR_BONUS = 50
KNIGHT_OUTPOST_BONUS = 25
ROOK_OPEN_FILE_BONUS = 25
ROOK_SEMI_OPEN_FILE_BONUS = 15
ROOK_ON_SEVENTH_BONUS = 30
QUEEN_ON_SEVENTH_BONUS = 15
CONNECTED_ROOKS_BONUS = 10

# Mobility weights (per legal move)
MOBILITY_WEIGHTS = {
    chess.KNIGHT: 4,
    chess.BISHOP: 5,
    chess.ROOK: 3,
    chess.QUEEN: 2,
}

# Space and development
SPACE_BONUS = 3  # Per square controlled in opponent's half
DEVELOPMENT_BONUS = 15  # Per developed minor piece
UNDEVELOPED_PENALTY = -20  # For pieces on back rank in opening
CASTLING_BONUS = 60
LOST_CASTLING_PENALTY = -40

# Threats
HANGING_PIECE_PENALTY = -50
ATTACK_ON_QUEEN_BONUS = 20
ATTACK_ON_KING_BONUS = 30


def flip_square(square: int) -> int:
    """Flip a square vertically for black's perspective."""
    return square ^ 56


def get_piece_value(piece_type: int) -> int:
    """Get the value of a piece type."""
    return PIECE_VALUES.get(piece_type, 0)


def get_piece_square_value(piece_type: int, square: int, 
                           color: bool, is_endgame: bool = False) -> int:
    """Get the positional value of a piece on a square."""
    if is_endgame:
        table = EG_TABLES.get(piece_type, [0] * 64)
    else:
        table = MG_TABLES.get(piece_type, [0] * 64)
    
    idx = square if color else flip_square(square)
    return table[idx]


# =============================================================================
# GAME PHASE CALCULATION
# =============================================================================

def calculate_game_phase(board: chess.Board) -> int:
    """Calculate game phase (0 = endgame, 256 = opening).
    
    Uses material to determine phase for tapered evaluation.
    """
    phase = 0
    
    # Phase weights: N=1, B=1, R=2, Q=4
    phase += len(board.pieces(chess.KNIGHT, chess.WHITE)) * 1
    phase += len(board.pieces(chess.KNIGHT, chess.BLACK)) * 1
    phase += len(board.pieces(chess.BISHOP, chess.WHITE)) * 1
    phase += len(board.pieces(chess.BISHOP, chess.BLACK)) * 1
    phase += len(board.pieces(chess.ROOK, chess.WHITE)) * 2
    phase += len(board.pieces(chess.ROOK, chess.BLACK)) * 2
    phase += len(board.pieces(chess.QUEEN, chess.WHITE)) * 4
    phase += len(board.pieces(chess.QUEEN, chess.BLACK)) * 4
    
    # Total phase at start = 4*1 + 4*1 + 4*2 + 2*4 = 24
    # Scale to 0-256
    return min(phase * 256 // 24, 256)


def is_endgame(board: chess.Board) -> bool:
    """Check if position is in endgame."""
    return calculate_game_phase(board) < 64


def count_material(board: chess.Board, color: bool) -> int:
    """Count total material for a color (excluding king)."""
    total = 0
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                       chess.ROOK, chess.QUEEN]:
        count = len(board.pieces(piece_type, color))
        total += count * MG_PIECE_VALUES[piece_type]
    return total


def evaluate_material(board: chess.Board) -> int:
    """Evaluate material balance (positive = white advantage)."""
    return count_material(board, chess.WHITE) - count_material(board, chess.BLACK)


# =============================================================================
# PAWN STRUCTURE EVALUATION
# =============================================================================

def evaluate_pawns(board: chess.Board, phase: int) -> int:
    """Comprehensive pawn structure evaluation."""
    mg_eval = 0
    eg_eval = 0
    
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    
    # Analyze white pawns
    white_pawn_files = [0] * 8
    for sq in white_pawns:
        white_pawn_files[chess.square_file(sq)] += 1
    
    # Analyze black pawns
    black_pawn_files = [0] * 8
    for sq in black_pawns:
        black_pawn_files[chess.square_file(sq)] += 1
    
    # White pawn evaluation
    for sq in white_pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        
        # Doubled pawns
        if white_pawn_files[file] > 1:
            mg_eval += DOUBLED_PAWN_PENALTY
            eg_eval += DOUBLED_PAWN_PENALTY
        
        # Isolated pawns (no friendly pawns on adjacent files)
        left_file = white_pawn_files[file - 1] if file > 0 else 0
        right_file = white_pawn_files[file + 1] if file < 7 else 0
        if left_file == 0 and right_file == 0:
            mg_eval += ISOLATED_PAWN_PENALTY
            eg_eval += ISOLATED_PAWN_PENALTY * 2  # More severe in endgame
        
        # Passed pawns (no enemy pawns blocking or on adjacent files ahead)
        is_passed = True
        for check_rank in range(rank + 1, 8):
            for check_file in range(max(0, file - 1), min(8, file + 2)):
                check_sq = chess.square(check_file, check_rank)
                if check_sq in black_pawns:
                    is_passed = False
                    break
            if not is_passed:
                break
        
        if is_passed:
            bonus = PASSED_PAWN_BONUS[rank]
            mg_eval += bonus // 2
            eg_eval += bonus  # Much more valuable in endgame
            
            # Connected passed pawns
            for adj_file in [file - 1, file + 1]:
                if 0 <= adj_file < 8:
                    adj_sq = chess.square(adj_file, rank)
                    if adj_sq in white_pawns:
                        eg_eval += CONNECTED_PASSED_BONUS
            
            # Protected passed pawn
            protect_rank = rank - 1
            if protect_rank >= 0:
                for prot_file in [file - 1, file + 1]:
                    if 0 <= prot_file < 8:
                        prot_sq = chess.square(prot_file, protect_rank)
                        if prot_sq in white_pawns:
                            eg_eval += PROTECTED_PASSED_BONUS
                            break
        
        # Backward pawns
        else:
            is_backward = True
            # Check if pawn can be supported by adjacent pawns
            for check_rank in range(0, rank):
                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file < 8:
                        check_sq = chess.square(adj_file, check_rank)
                        if check_sq in white_pawns:
                            is_backward = False
                            break
                if not is_backward:
                    break
            
            # Only backward if the square in front is controlled by enemy pawn
            if is_backward and rank < 7:
                front_sq = chess.square(file, rank + 1)
                front_attacked = False
                for att_file in [file - 1, file + 1]:
                    if 0 <= att_file < 8:
                        att_sq = chess.square(att_file, rank + 2) if rank + 2 < 8 else None
                        if att_sq and att_sq in black_pawns:
                            front_attacked = True
                            break
                if front_attacked:
                    mg_eval += BACKWARD_PAWN_PENALTY
                    eg_eval += BACKWARD_PAWN_PENALTY
    
    # Black pawn evaluation (mirror of white)
    for sq in black_pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        
        # Doubled pawns
        if black_pawn_files[file] > 1:
            mg_eval -= DOUBLED_PAWN_PENALTY
            eg_eval -= DOUBLED_PAWN_PENALTY
        
        # Isolated pawns
        left_file = black_pawn_files[file - 1] if file > 0 else 0
        right_file = black_pawn_files[file + 1] if file < 7 else 0
        if left_file == 0 and right_file == 0:
            mg_eval -= ISOLATED_PAWN_PENALTY
            eg_eval -= ISOLATED_PAWN_PENALTY * 2
        
        # Passed pawns
        is_passed = True
        for check_rank in range(0, rank):
            for check_file in range(max(0, file - 1), min(8, file + 2)):
                check_sq = chess.square(check_file, check_rank)
                if check_sq in white_pawns:
                    is_passed = False
                    break
            if not is_passed:
                break
        
        if is_passed:
            bonus = PASSED_PAWN_BONUS[7 - rank]  # Flip rank for black
            mg_eval -= bonus // 2
            eg_eval -= bonus
            
            # Connected passed pawns
            for adj_file in [file - 1, file + 1]:
                if 0 <= adj_file < 8:
                    adj_sq = chess.square(adj_file, rank)
                    if adj_sq in black_pawns:
                        eg_eval -= CONNECTED_PASSED_BONUS
            
            # Protected passed pawn
            protect_rank = rank + 1
            if protect_rank < 8:
                for prot_file in [file - 1, file + 1]:
                    if 0 <= prot_file < 8:
                        prot_sq = chess.square(prot_file, protect_rank)
                        if prot_sq in black_pawns:
                            eg_eval -= PROTECTED_PASSED_BONUS
                            break
    
    # Tapered evaluation
    return (mg_eval * phase + eg_eval * (256 - phase)) // 256


# =============================================================================
# KING SAFETY EVALUATION
# =============================================================================

# Attack weights by piece type (for king danger calculation)
PIECE_ATTACK_WEIGHT = {
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 5,
}

# Safety table: maps attack units to penalty (exponential danger)
SAFETY_TABLE = [
    0, 0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30, 35, 40,
    46, 52, 58, 65, 72, 80, 88, 97, 106, 116, 126, 137, 149, 161, 174, 188,
    202, 217, 233, 250, 268, 287, 307, 328, 350, 373, 397, 422, 448, 476, 505, 535,
    566, 598, 632, 667, 703, 741, 780, 821, 863, 907, 952, 999, 999, 999, 999, 999
]


def evaluate_king_safety(board: chess.Board, phase: int) -> int:
    """Evaluate king safety (much more important in middlegame)."""
    if phase < 32:  # Deep endgame - king safety barely matters
        return 0
    
    mg_eval = 0
    
    # White king safety
    white_king_sq = board.king(chess.WHITE)
    if white_king_sq is not None:
        mg_eval += _evaluate_king_zone(board, white_king_sq, chess.WHITE)
    
    # Black king safety
    black_king_sq = board.king(chess.BLACK)
    if black_king_sq is not None:
        mg_eval -= _evaluate_king_zone(board, black_king_sq, chess.BLACK)
    
    # Scale by game phase (king safety matters much more in middlegame)
    return (mg_eval * phase) // 256


def _get_king_zone(king_sq: int, color: bool) -> list:
    """Get extended king zone (3x4 area in front of king + surrounding squares)."""
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    zone = []
    
    # Direction towards enemy
    forward = 1 if color == chess.WHITE else -1
    
    # Core zone: 3x3 around king
    for dr in range(-1, 2):
        for df in range(-1, 2):
            r, f = king_rank + dr, king_file + df
            if 0 <= r < 8 and 0 <= f < 8:
                zone.append(chess.square(f, r))
    
    # Extended zone: 3 squares two ranks in front
    extended_rank = king_rank + 2 * forward
    if 0 <= extended_rank < 8:
        for df in range(-1, 2):
            f = king_file + df
            if 0 <= f < 8:
                zone.append(chess.square(f, extended_rank))
    
    return zone


def _evaluate_pawn_shield(board: chess.Board, king_sq: int, color: bool) -> int:
    """Evaluate pawn shield strength in front of king."""
    shield_score = 0
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    
    # Only evaluate if king is on back ranks (castled position)
    home_rank = 0 if color == chess.WHITE else 7
    if abs(king_rank - home_rank) > 1:
        return -30  # King not on safe rank - penalty
    
    pawn_direction = 1 if color == chess.WHITE else -1
    our_pawns = board.pieces(chess.PAWN, color)
    
    # Check three files around king
    for file_offset in [-1, 0, 1]:
        f = king_file + file_offset
        if not (0 <= f < 8):
            continue
        
        # Ideal pawn positions (closest to king)
        ideal_rank = king_rank + pawn_direction
        advanced_rank = king_rank + 2 * pawn_direction
        
        pawn_found = False
        
        # Check ideal position first
        if 0 <= ideal_rank < 8:
            sq = chess.square(f, ideal_rank)
            if sq in our_pawns:
                if file_offset == 0:
                    shield_score += 20  # Pawn directly in front
                else:
                    shield_score += 15  # Pawn on adjacent file
                pawn_found = True
        
        # Check one rank further if not found
        if not pawn_found and 0 <= advanced_rank < 8:
            sq = chess.square(f, advanced_rank)
            if sq in our_pawns:
                if file_offset == 0:
                    shield_score += 10  # Advanced pawn in front
                else:
                    shield_score += 8   # Advanced pawn on adjacent file
                pawn_found = True
        
        # Penalty for missing pawn shield
        if not pawn_found:
            if file_offset == 0:
                shield_score -= 25  # Missing pawn directly in front is worst
            else:
                shield_score -= 15  # Missing flank pawn
    
    return shield_score


def _evaluate_king_zone(board: chess.Board, king_sq: int, color: bool) -> int:
    """Comprehensive evaluation of king safety."""
    eval_score = 0
    enemy_color = not color
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    
    # =========================================================================
    # 1. PAWN SHIELD EVALUATION
    # =========================================================================
    eval_score += _evaluate_pawn_shield(board, king_sq, color)
    
    # =========================================================================
    # 2. CASTLING STATUS
    # =========================================================================
    home_rank = 0 if color == chess.WHITE else 7
    
    # Bonus for castled king position
    if king_rank == home_rank:
        if king_file >= 6:  # Kingside castled (g1/h1 or g8/h8)
            eval_score += 45
        elif king_file <= 2:  # Queenside castled (a1/b1/c1 or a8/b8/c8)
            eval_score += 35
        elif king_file in [3, 4]:  # King still in center
            # Count major pieces to assess danger
            enemy_queens = len(board.pieces(chess.QUEEN, enemy_color))
            enemy_rooks = len(board.pieces(chess.ROOK, enemy_color))
            if enemy_queens > 0:
                eval_score -= 60  # Very dangerous with queen on board
            elif enemy_rooks > 0:
                eval_score -= 30  # Somewhat dangerous
    
    # =========================================================================
    # 3. OPEN/SEMI-OPEN FILES NEAR KING
    # =========================================================================
    our_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, enemy_color)
    
    for f in range(max(0, king_file - 1), min(8, king_file + 2)):
        file_mask = chess.BB_FILES[f]
        has_our_pawn = bool(our_pawns & file_mask)
        has_enemy_pawn = bool(enemy_pawns & file_mask)
        
        if not has_our_pawn and not has_enemy_pawn:
            eval_score += KING_OPEN_FILE_PENALTY  # -25: Open file
            # Extra penalty if enemy rook/queen on this file
            for piece_type in [chess.ROOK, chess.QUEEN]:
                for sq in board.pieces(piece_type, enemy_color):
                    if chess.square_file(sq) == f:
                        eval_score -= 20
                        break
        elif not has_our_pawn:
            eval_score += KING_SEMI_OPEN_FILE_PENALTY  # -15: Semi-open file
    
    # =========================================================================
    # 4. PAWN STORM DETECTION (enemy pawns advancing toward our king)
    # =========================================================================
    storm_penalty = 0
    for f in range(max(0, king_file - 1), min(8, king_file + 2)):
        for sq in enemy_pawns:
            if chess.square_file(sq) == f:
                pawn_rank = chess.square_rank(sq)
                # Distance from our king's rank
                if color == chess.WHITE:
                    advancement = pawn_rank  # Higher rank = more advanced
                    if advancement >= 4:  # Pawn on 5th rank or beyond
                        storm_penalty += (advancement - 3) * 8
                else:
                    advancement = 7 - pawn_rank
                    if advancement >= 4:
                        storm_penalty += (advancement - 3) * 8
    eval_score -= storm_penalty
    
    # =========================================================================
    # 5. KING ATTACKERS (main king safety component)
    # =========================================================================
    king_zone = _get_king_zone(king_sq, color)
    attack_units = 0
    attacker_count = 0
    
    # Track which pieces are attacking the king zone
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == enemy_color and piece.piece_type != chess.KING:
            attacks = board.attacks(sq)
            zone_attacks = sum(1 for z in king_zone if z in attacks)
            
            if zone_attacks > 0:
                attacker_count += 1
                piece_weight = PIECE_ATTACK_WEIGHT.get(piece.piece_type, 1)
                attack_units += zone_attacks * piece_weight
    
    # Apply non-linear safety penalty (more attackers = exponentially worse)
    if attacker_count >= 2:
        safety_idx = min(attack_units, 63)
        eval_score -= SAFETY_TABLE[safety_idx]
    
    # =========================================================================
    # 6. QUEEN TROPISM (enemy queen close to king is dangerous)
    # =========================================================================
    for queen_sq in board.pieces(chess.QUEEN, enemy_color):
        distance = chess.square_distance(king_sq, queen_sq)
        if distance <= 4:
            eval_score -= (5 - distance) * 15  # Up to 60 penalty for adjacent queen
    
    # =========================================================================
    # 7. WEAK SQUARES AROUND KING (squares not defended by pawns)
    # =========================================================================
    weak_squares = 0
    # Calculate pawn attack squares
    pawn_attack_squares = set()
    for pawn_sq in our_pawns:
        # Pawns attack diagonally
        pawn_file = chess.square_file(pawn_sq)
        pawn_rank = chess.square_rank(pawn_sq)
        attack_rank = pawn_rank + (1 if color == chess.WHITE else -1)
        if 0 <= attack_rank <= 7:
            if pawn_file > 0:
                pawn_attack_squares.add(chess.square(pawn_file - 1, attack_rank))
            if pawn_file < 7:
                pawn_attack_squares.add(chess.square(pawn_file + 1, attack_rank))
    
    for zone_sq in king_zone:
        if zone_sq != king_sq and zone_sq not in pawn_attack_squares:
            # Check if enemy can attack this weak square
            if board.attackers(enemy_color, zone_sq):
                weak_squares += 1
    
    eval_score -= weak_squares * 8
    
    # =========================================================================
    # 8. KING DEFENDERS (pieces defending king zone)
    # =========================================================================
    defender_count = 0
    for zone_sq in king_zone:
        defenders = board.attackers(color, zone_sq)
        # Count non-king defenders (SquareSet is iterable)
        for def_sq in defenders:
            piece = board.piece_at(def_sq)
            if piece and piece.piece_type != chess.KING:
                defender_count += 1
                break  # Count each square once
    
    eval_score += min(defender_count, 4) * 5  # Bonus for defenders (up to 20)
    
    # =========================================================================
    # 9. CHECK VULNERABILITY (can enemy give check easily?)
    # =========================================================================
    if board.is_check():
        eval_score -= 20  # Currently in check is bad
    
    return eval_score


# =============================================================================
# PIECE EVALUATION
# =============================================================================

def evaluate_pieces(board: chess.Board, phase: int) -> int:
    """Evaluate piece placement and coordination."""
    mg_eval = 0
    eg_eval = 0
    
    # Bishop pair
    white_bishops = list(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = list(board.pieces(chess.BISHOP, chess.BLACK))
    
    if len(white_bishops) >= 2:
        mg_eval += BISHOP_PAIR_BONUS
        eg_eval += BISHOP_PAIR_BONUS + 20  # Even more valuable in endgame
    if len(black_bishops) >= 2:
        mg_eval -= BISHOP_PAIR_BONUS
        eg_eval -= BISHOP_PAIR_BONUS + 20
    
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    
    # Knight outposts (knight on 4th-6th rank, protected by pawn, can't be attacked by enemy pawn)
    for sq in board.pieces(chess.KNIGHT, chess.WHITE):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        
        if 3 <= rank <= 5:  # 4th to 6th rank
            # Check if protected by pawn
            is_protected = False
            for prot_file in [file - 1, file + 1]:
                if 0 <= prot_file < 8:
                    prot_sq = chess.square(prot_file, rank - 1)
                    if prot_sq in white_pawns:
                        is_protected = True
                        break
            
            # Check if can be attacked by enemy pawn
            can_be_attacked = False
            for att_file in [file - 1, file + 1]:
                if 0 <= att_file < 8:
                    for att_rank in range(rank + 1, 8):
                        att_sq = chess.square(att_file, att_rank)
                        if att_sq in black_pawns:
                            can_be_attacked = True
                            break
                    if can_be_attacked:
                        break
            
            if is_protected and not can_be_attacked:
                mg_eval += KNIGHT_OUTPOST_BONUS
                eg_eval += KNIGHT_OUTPOST_BONUS // 2
    
    # Black knight outposts
    for sq in board.pieces(chess.KNIGHT, chess.BLACK):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        
        if 2 <= rank <= 4:  # 3rd to 5th rank for black
            is_protected = False
            for prot_file in [file - 1, file + 1]:
                if 0 <= prot_file < 8:
                    prot_sq = chess.square(prot_file, rank + 1)
                    if prot_sq in black_pawns:
                        is_protected = True
                        break
            
            can_be_attacked = False
            for att_file in [file - 1, file + 1]:
                if 0 <= att_file < 8:
                    for att_rank in range(0, rank):
                        att_sq = chess.square(att_file, att_rank)
                        if att_sq in white_pawns:
                            can_be_attacked = True
                            break
                    if can_be_attacked:
                        break
            
            if is_protected and not can_be_attacked:
                mg_eval -= KNIGHT_OUTPOST_BONUS
                eg_eval -= KNIGHT_OUTPOST_BONUS // 2
    
    # Rook evaluation
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        
        # Open/semi-open files
        white_pawn_on_file = any(chess.square(file, r) in white_pawns for r in range(8))
        black_pawn_on_file = any(chess.square(file, r) in black_pawns for r in range(8))
        
        if not white_pawn_on_file and not black_pawn_on_file:
            mg_eval += ROOK_OPEN_FILE_BONUS
            eg_eval += ROOK_OPEN_FILE_BONUS
        elif not white_pawn_on_file:
            mg_eval += ROOK_SEMI_OPEN_FILE_BONUS
            eg_eval += ROOK_SEMI_OPEN_FILE_BONUS
        
        # Rook on 7th rank
        if rank == 6:  # 7th rank for white
            mg_eval += ROOK_ON_SEVENTH_BONUS
            eg_eval += ROOK_ON_SEVENTH_BONUS + 10
    
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        
        white_pawn_on_file = any(chess.square(file, r) in white_pawns for r in range(8))
        black_pawn_on_file = any(chess.square(file, r) in black_pawns for r in range(8))
        
        if not white_pawn_on_file and not black_pawn_on_file:
            mg_eval -= ROOK_OPEN_FILE_BONUS
            eg_eval -= ROOK_OPEN_FILE_BONUS
        elif not black_pawn_on_file:
            mg_eval -= ROOK_SEMI_OPEN_FILE_BONUS
            eg_eval -= ROOK_SEMI_OPEN_FILE_BONUS
        
        if rank == 1:  # 2nd rank for black (7th rank)
            mg_eval -= ROOK_ON_SEVENTH_BONUS
            eg_eval -= ROOK_ON_SEVENTH_BONUS + 10
    
    # Connected rooks
    white_rooks = list(board.pieces(chess.ROOK, chess.WHITE))
    if len(white_rooks) == 2:
        r1, r2 = white_rooks
        # Check if they can see each other (same rank or file, no pieces between)
        if chess.square_rank(r1) == chess.square_rank(r2) or chess.square_file(r1) == chess.square_file(r2):
            # Simplified: just check if they attack each other
            if r2 in board.attacks(r1):
                mg_eval += CONNECTED_ROOKS_BONUS
                eg_eval += CONNECTED_ROOKS_BONUS
    
    black_rooks = list(board.pieces(chess.ROOK, chess.BLACK))
    if len(black_rooks) == 2:
        r1, r2 = black_rooks
        if chess.square_rank(r1) == chess.square_rank(r2) or chess.square_file(r1) == chess.square_file(r2):
            if r2 in board.attacks(r1):
                mg_eval -= CONNECTED_ROOKS_BONUS
                eg_eval -= CONNECTED_ROOKS_BONUS
    
    # Queen on 7th
    for sq in board.pieces(chess.QUEEN, chess.WHITE):
        if chess.square_rank(sq) == 6:
            mg_eval += QUEEN_ON_SEVENTH_BONUS
    for sq in board.pieces(chess.QUEEN, chess.BLACK):
        if chess.square_rank(sq) == 1:
            mg_eval -= QUEEN_ON_SEVENTH_BONUS
    
    return (mg_eval * phase + eg_eval * (256 - phase)) // 256


# =============================================================================
# MOBILITY EVALUATION
# =============================================================================

def evaluate_mobility(board: chess.Board) -> int:
    """Evaluate piece mobility."""
    eval_score = 0
    
    # Count legal moves for each piece type
    original_turn = board.turn
    
    for color in [chess.WHITE, chess.BLACK]:
        board.turn = color
        multiplier = 1 if color == chess.WHITE else -1
        
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type in MOBILITY_WEIGHTS:
                eval_score += multiplier * MOBILITY_WEIGHTS[piece.piece_type]
    
    board.turn = original_turn
    return eval_score


# =============================================================================
# DEVELOPMENT AND CASTLING
# =============================================================================

def evaluate_development(board: chess.Board, phase: int) -> int:
    """Evaluate development (important in opening/early middlegame)."""
    if phase < 128:  # Less relevant in endgame
        return 0
    
    eval_score = 0
    
    # White development
    # Knights not on b1/g1
    if board.piece_at(chess.B1) != chess.Piece(chess.KNIGHT, chess.WHITE):
        eval_score += DEVELOPMENT_BONUS // 2
    else:
        eval_score += UNDEVELOPED_PENALTY
    if board.piece_at(chess.G1) != chess.Piece(chess.KNIGHT, chess.WHITE):
        eval_score += DEVELOPMENT_BONUS // 2
    else:
        eval_score += UNDEVELOPED_PENALTY
    
    # Bishops not on c1/f1
    if board.piece_at(chess.C1) != chess.Piece(chess.BISHOP, chess.WHITE):
        eval_score += DEVELOPMENT_BONUS // 2
    else:
        eval_score += UNDEVELOPED_PENALTY
    if board.piece_at(chess.F1) != chess.Piece(chess.BISHOP, chess.WHITE):
        eval_score += DEVELOPMENT_BONUS // 2
    else:
        eval_score += UNDEVELOPED_PENALTY
    
    # Black development
    if board.piece_at(chess.B8) != chess.Piece(chess.KNIGHT, chess.BLACK):
        eval_score -= DEVELOPMENT_BONUS // 2
    else:
        eval_score -= UNDEVELOPED_PENALTY
    if board.piece_at(chess.G8) != chess.Piece(chess.KNIGHT, chess.BLACK):
        eval_score -= DEVELOPMENT_BONUS // 2
    else:
        eval_score -= UNDEVELOPED_PENALTY
    
    if board.piece_at(chess.C8) != chess.Piece(chess.BISHOP, chess.BLACK):
        eval_score -= DEVELOPMENT_BONUS // 2
    else:
        eval_score -= UNDEVELOPED_PENALTY
    if board.piece_at(chess.F8) != chess.Piece(chess.BISHOP, chess.BLACK):
        eval_score -= DEVELOPMENT_BONUS // 2
    else:
        eval_score -= UNDEVELOPED_PENALTY
    
    # Castling rights and status
    white_king_sq = board.king(chess.WHITE)
    black_king_sq = board.king(chess.BLACK)
    
    # Bonus for having castled (king on castled squares)
    if white_king_sq in [chess.G1, chess.C1]:
        eval_score += CASTLING_BONUS
    elif white_king_sq == chess.E1:
        # King still on e1
        if board.has_castling_rights(chess.WHITE):
            # Still has some castling rights - small bonus for retaining option
            eval_score += 10
        else:
            # Lost castling rights without castling (rooks moved or king touched)
            eval_score += LOST_CASTLING_PENALTY
    else:
        # King moved somewhere else (not castling squares)
        if not board.has_castling_rights(chess.WHITE):
            eval_score += LOST_CASTLING_PENALTY
    
    if black_king_sq in [chess.G8, chess.C8]:
        eval_score -= CASTLING_BONUS
    elif black_king_sq == chess.E8:
        if board.has_castling_rights(chess.BLACK):
            eval_score -= 10  # Black retains castling option
        else:
            eval_score -= LOST_CASTLING_PENALTY
    else:
        if not board.has_castling_rights(chess.BLACK):
            eval_score -= LOST_CASTLING_PENALTY
    
    return (eval_score * phase) // 256


# =============================================================================
# SPACE EVALUATION
# =============================================================================

def evaluate_space(board: chess.Board, phase: int) -> int:
    """Evaluate space advantage (control of squares in opponent's half)."""
    if phase < 64:  # Less relevant in endgame
        return 0
    
    eval_score = 0
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    
    # Count white's space (squares on ranks 4-6 behind pawns)
    for sq in white_pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        if rank >= 3:  # Pawn on 4th rank or beyond
            # Count squares behind pawn
            for r in range(3, rank):
                eval_score += SPACE_BONUS
    
    # Count black's space
    for sq in black_pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        if rank <= 4:  # Pawn on 5th rank or beyond (for black)
            for r in range(rank + 1, 5):
                eval_score -= SPACE_BONUS
    
    return (eval_score * phase) // 256


# =============================================================================
# PIECE-SQUARE TABLE EVALUATION
# =============================================================================

def evaluate_psqt(board: chess.Board, phase: int) -> int:
    """Evaluate piece-square tables with tapered evaluation."""
    mg_eval = 0
    eg_eval = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        piece_type = piece.piece_type
        color = piece.color
        
        # Get PST value
        idx = square if color == chess.WHITE else flip_square(square)
        mg_val = MG_TABLES[piece_type][idx]
        eg_val = EG_TABLES[piece_type][idx]
        
        if color == chess.WHITE:
            mg_eval += mg_val
            eg_eval += eg_val
        else:
            mg_eval -= mg_val
            eg_eval -= eg_val
    
    return (mg_eval * phase + eg_eval * (256 - phase)) // 256


# =============================================================================
# THREATS EVALUATION
# =============================================================================

def evaluate_threats(board: chess.Board) -> int:
    """Evaluate tactical threats."""
    eval_score = 0
    
    # Check for hanging pieces (attacked by lower value piece or undefended)
    for color in [chess.WHITE, chess.BLACK]:
        multiplier = 1 if color == chess.WHITE else -1
        enemy = not color
        
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            for sq in board.pieces(piece_type, color):
                attackers = board.attackers(enemy, sq)
                defenders = board.attackers(color, sq)
                
                if attackers and not defenders:
                    # Hanging piece
                    eval_score -= multiplier * (PIECE_VALUES[piece_type] // 4)
                elif attackers:
                    # Check if attacked by lower value piece
                    min_attacker_value = min(
                        PIECE_VALUES.get(board.piece_at(att_sq).piece_type, 10000)
                        for att_sq in attackers
                    )
                    if min_attacker_value < PIECE_VALUES[piece_type]:
                        eval_score -= multiplier * 15  # Attacked by lower value
    
    return eval_score


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_position(board: chess.Board, is_endgame_phase: bool = None) -> int:
    """Full position evaluation in centipawns.
    
    Returns positive value for white advantage, negative for black.
    Uses tapered evaluation to smoothly transition between middlegame and endgame.
    """
    # Handle terminal positions
    if board.is_checkmate():
        return -30000 if board.turn else 30000
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    if board.can_claim_draw():
        return 0
    
    # Calculate game phase for tapered evaluation
    phase = calculate_game_phase(board)
    
    evaluation = 0
    
    # Material (most important)
    evaluation += evaluate_material(board)
    
    # Piece-square tables with tapered eval
    evaluation += evaluate_psqt(board, phase)
    
    # Pawn structure
    evaluation += evaluate_pawns(board, phase)
    
    # King safety
    evaluation += evaluate_king_safety(board, phase)
    
    # Piece evaluation (bishop pair, outposts, rooks, etc.)
    evaluation += evaluate_pieces(board, phase)
    
    # Mobility
    evaluation += evaluate_mobility(board)
    
    # Development (opening/middlegame)
    evaluation += evaluate_development(board, phase)
    
    # Space advantage
    evaluation += evaluate_space(board, phase)
    
    # Threats
    evaluation += evaluate_threats(board)
    
    # Tempo bonus (small bonus for side to move)
    if board.turn == chess.WHITE:
        evaluation += 10
    else:
        evaluation -= 10
    
    return evaluation


# =============================================================================
# UTILITY FUNCTIONS (for compatibility)
# =============================================================================

def evaluate_center_control(board: chess.Board) -> int:
    """Evaluate control of central squares (legacy function)."""
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    
    evaluation = 0
    for sq in center_squares:
        piece = board.piece_at(sq)
        if piece:
            bonus = 15
            if piece.color == chess.WHITE:
                evaluation += bonus
            else:
                evaluation -= bonus
    
    return evaluation


def evaluate_bishop_pair(board: chess.Board) -> int:
    """Bonus for having the bishop pair (legacy function)."""
    evaluation = 0
    
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        evaluation += BISHOP_PAIR_BONUS
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        evaluation -= BISHOP_PAIR_BONUS
    
    return evaluation


def evaluate_rook_on_open_file(board: chess.Board) -> int:
    """Bonus for rooks on open files (legacy function)."""
    evaluation = 0
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        file = chess.square_file(sq)
        white_pawn_on_file = any(chess.square(file, r) in white_pawns for r in range(8))
        black_pawn_on_file = any(chess.square(file, r) in black_pawns for r in range(8))
        
        if not white_pawn_on_file and not black_pawn_on_file:
            evaluation += ROOK_OPEN_FILE_BONUS
        elif not white_pawn_on_file:
            evaluation += ROOK_SEMI_OPEN_FILE_BONUS
    
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        file = chess.square_file(sq)
        white_pawn_on_file = any(chess.square(file, r) in white_pawns for r in range(8))
        black_pawn_on_file = any(chess.square(file, r) in black_pawns for r in range(8))
        
        if not white_pawn_on_file and not black_pawn_on_file:
            evaluation -= ROOK_OPEN_FILE_BONUS
        elif not black_pawn_on_file:
            evaluation -= ROOK_SEMI_OPEN_FILE_BONUS
    
    return evaluation


def get_piece_name(piece_type: int) -> str:
    """Get the name of a piece type."""
    names = {
        chess.PAWN: 'Pawn',
        chess.KNIGHT: 'Knight',
        chess.BISHOP: 'Bishop',
        chess.ROOK: 'Rook',
        chess.QUEEN: 'Queen',
        chess.KING: 'King',
    }
    return names.get(piece_type, 'Unknown')
