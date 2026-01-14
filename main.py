"""
Main entry point for the Chess Engine.
Provides a simple command-line interface to play against the engine.
"""

import chess
import os
from src.engine import ChessEngine


def main():
    """Main function to run the chess engine."""
    print("=" * 50)
    print("    ML Chess Engine - MCTS + PyTorch")
    print("=" * 50)
    print()
    
    # Load trained model if available
    model_path = os.path.join(os.path.dirname(__file__), "models", "chess_model.pt")
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
    else:
        print("WARNING: No trained model found! Using random neural network.")
        print("         Run train.py first to train the model.")
        model_path = None
    
    # Initialize engine with trained model and reasonable simulation count
    print("Initializing neural network...")
    engine = ChessEngine(model_path=model_path, num_simulations=400)
    print("Engine ready!")
    print()
    
    print("Commands:")
    print("  - Enter moves in UCI format (e.g., e2e4, g1f3)")
    print("  - Castling: O-O or 0-0 (kingside), O-O-O or 0-0-0 (queenside)")
    print("  - Type 'quit' to exit")
    print("  - Type 'new' to start a new game")
    print("  - Type 'fen' to show current position FEN")
    print("  - Type 'moves' to show legal moves")
    print("  - Type 'auto' to let engine play both sides")
    print()
    
    while True:
        # Display board
        print(engine.display())
        print()
        
        if engine.is_game_over():
            result = engine.get_result()
            print(f"Game Over! Result: {result}")
            user_input = input("Type 'new' for new game or 'quit' to exit: ").strip().lower()
            if user_input == 'new':
                engine.new_game()
                continue
            elif user_input == 'quit':
                break
            continue
        
        # Show whose turn it is
        turn = "White" if engine.board.turn else "Black"
        print(f"{turn} to move")
        
        user_input = input("Your move (or command): ").strip().lower()
        
        if user_input == 'quit':
            print("Thanks for playing!")
            break
        
        elif user_input == 'new':
            engine.new_game()
            print("New game started!")
            continue
        
        elif user_input == 'fen':
            print(f"FEN: {engine.get_fen()}")
            continue
        
        elif user_input == 'moves':
            moves = [m.uci() for m in engine.get_legal_moves()]
            print(f"Legal moves: {', '.join(moves)}")
            continue
        
        elif user_input == 'auto':
            print("\nEngine thinking...")
            best_move, win_prob = engine.get_best_move()
            print(f"Engine plays: {best_move.uci()} (eval: {win_prob:.1%} win probability)")
            engine.make_move(best_move)
            continue
        
        elif user_input == 'hint':
            print("\nCalculating best move...")
            best_move, win_prob = engine.get_best_move()
            print(f"Suggested move: {best_move.uci()} (eval: {win_prob:.1%} win probability)")
            continue
        
        else:
            # Try to make the move
            if engine.make_move_uci(user_input):
                print(f"You played: {user_input}")
                
                # Engine's response
                if not engine.is_game_over():
                    print("\nEngine thinking...")
                    best_move, win_prob = engine.get_best_move()
                    print(f"Engine plays: {best_move.uci()} (eval: {win_prob:.1%} win probability)")
                    engine.make_move(best_move)
            else:
                print(f"Invalid move: {user_input}")
                print("Enter moves in UCI format (e.g., e2e4)")
        
        print()


if __name__ == "__main__":
    main()
