"""
Training script for the chess neural network.
"""

import argparse
import os
import time
from src.neural_network import ChessNeuralNetwork
from src.trainer import SelfPlayTrainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train the chess neural network')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of self-play games per iteration (default: 100)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of training iterations (default: 10)')
    parser.add_argument('--simulations', type=int, default=200,
                        help='MCTS simulations per move (default: 200)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Training epochs per iteration (default: 5)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Initial move selection temperature (default: 1.0)')
    parser.add_argument('--temp-threshold', type=int, default=30,
                        help='Move number after which temperature drops (default: 30)')
    parser.add_argument('--output', type=str, default='models/chess_model.pt',
                        help='Output model path (default: models/chess_model.pt)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load existing model to continue training')
    parser.add_argument('--checkpoint-freq', type=int, default=1,
                        help='Save checkpoint every N iterations (default: 1)')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation (default: True)')
    parser.add_argument('--replay-size', type=int, default=50000,
                        help='Maximum replay buffer size (default: 50000)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--lr-decay', type=float, default=0.95,
                        help='Learning rate decay per iteration (default: 0.95)')
    parser.add_argument('--log-games', action='store_true', default=True,
                        help='Log games to PGN files (default: True)')
    parser.add_argument('--no-log-games', action='store_false', dest='log_games',
                        help='Disable game logging')
    parser.add_argument('--log-dir', type=str, default='games',
                        help='Directory for game logs (default: games)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("       Chess Neural Network Training (AlphaZero-style)")
    print("=" * 60)
    print()
    print(f"Configuration:")
    print(f"  - Training iterations:    {args.iterations}")
    print(f"  - Self-play games/iter:   {args.games}")
    print(f"  - MCTS simulations:       {args.simulations}")
    print(f"  - Training epochs/iter:   {args.epochs}")
    print(f"  - Batch size:             {args.batch_size}")
    print(f"  - Initial temperature:    {args.temperature}")
    print(f"  - Temperature threshold:  {args.temp_threshold} moves")
    print(f"  - Learning rate:          {args.lr}")
    print(f"  - LR decay:               {args.lr_decay}")
    print(f"  - Replay buffer size:     {args.replay_size}")
    print(f"  - Data augmentation:      {args.augment}")
    print(f"  - Log games:              {args.log_games}")
    if args.log_games:
        print(f"  - Game log directory:     {args.log_dir}")
    print(f"  - Output model:           {args.output}")
    print()
    
    total_games = args.iterations * args.games
    print(f"  => Total games to play:   {total_games}")
    print()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Initialize neural network
    print("Initializing neural network...")
    if args.load:
        print(f"Loading existing model from {args.load}")
        nn = ChessNeuralNetwork(args.load)
    else:
        nn = ChessNeuralNetwork()
    
    # Set learning rate
    for param_group in nn.optimizer.param_groups:
        param_group['lr'] = args.lr
    
    nn.summary()
    print()
    
    # Initialize trainer
    trainer = SelfPlayTrainer(
        nn, 
        num_simulations=args.simulations,
        replay_buffer_size=args.replay_size,
        use_augmentation=args.augment,
        log_games=args.log_games,
        log_dir=args.log_dir
    )
    
    # Training loop
    start_time = time.time()
    
    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()
        print("=" * 60)
        print(f"  ITERATION {iteration}/{args.iterations}")
        print("=" * 60)
        
        # Decrease temperature over iterations for more exploitation later
        current_temp = args.temperature * (0.9 ** (iteration - 1))
        current_temp = max(current_temp, 0.3)  # Minimum temperature
        
        print(f"Current temperature: {current_temp:.2f}")
        print()
        
        # Run training iteration
        trainer.train_iteration(
            num_games=args.games,
            epochs=args.epochs,
            batch_size=args.batch_size,
            temperature=current_temp,
            temp_threshold=args.temp_threshold
        )
        
        # Decay learning rate
        for param_group in nn.optimizer.param_groups:
            param_group['lr'] *= args.lr_decay
        current_lr = nn.optimizer.param_groups[0]['lr']
        print(f"Learning rate decayed to: {current_lr:.6f}")
        
        # Save checkpoint
        if iteration % args.checkpoint_freq == 0:
            checkpoint_path = args.output.replace('.pt', f'_iter{iteration}.pt')
            print(f"Saving checkpoint to {checkpoint_path}...")
            nn.save(checkpoint_path)
        
        iter_time = time.time() - iter_start
        print(f"Iteration {iteration} completed in {iter_time/60:.1f} minutes")
        print()
    
    # Save final model
    print(f"\nSaving final model to {args.output}...")
    nn.save(args.output)
    
    total_time = time.time() - start_time
    print()
    print("=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total training time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Total games played: {total_games}")
    print(f"Final model saved to: {args.output}")


if __name__ == "__main__":
    main()
