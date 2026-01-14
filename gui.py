"""
Chess Engine GUI - Play and Train
Simple tkinter-based GUI for playing against the engine and training.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import chess
import threading
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.engine import ChessEngine
from src.neural_network import ChessNeuralNetwork
from src.trainer import SelfPlayTrainer


class ChessGUI:
    """Main GUI application for the chess engine."""
    
    # Unicode chess pieces
    PIECES = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    }
    
    LIGHT_SQUARE = '#F0D9B5'
    DARK_SQUARE = '#B58863'
    HIGHLIGHT = '#CDD26A'
    SELECTED = '#829769'
    
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Engine - ML + MCTS")
        self.root.geometry("900x700")
        self.root.configure(bg='#2C2C2C')
        
        # Engine state
        self.engine = None
        self.selected_square = None
        self.player_color = chess.WHITE
        self.engine_thinking = False
        self.training_thread = None
        self.training_active = False
        
        # Create main frames
        self.create_menu()
        self.create_main_layout()
        
        # Initialize engine
        self.init_engine()
    
    def create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Game menu
        game_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="New Game (Play as White)", command=lambda: self.new_game(chess.WHITE))
        game_menu.add_command(label="New Game (Play as Black)", command=lambda: self.new_game(chess.BLACK))
        game_menu.add_separator()
        game_menu.add_command(label="Flip Board", command=self.flip_board)
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.root.quit)
        
        # Model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(label="Load Model...", command=self.load_model)
        model_menu.add_command(label="Save Model...", command=self.save_model)
    
    def create_main_layout(self):
        """Create main application layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Chess board
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Board canvas
        self.board_frame = ttk.Frame(left_frame)
        self.board_frame.pack(pady=10)
        
        self.square_size = 65
        self.canvas = tk.Canvas(
            self.board_frame, 
            width=self.square_size * 8, 
            height=self.square_size * 8,
            highlightthickness=2,
            highlightbackground='#1a1a1a'
        )
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.on_square_click)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(left_frame, textvariable=self.status_var, font=('Segoe UI', 12))
        self.status_label.pack(pady=5)
        
        # Eval bar
        eval_frame = ttk.Frame(left_frame)
        eval_frame.pack(fill=tk.X, pady=5)
        ttk.Label(eval_frame, text="Engine Eval:").pack(side=tk.LEFT)
        self.eval_var = tk.StringVar(value="0.0")
        ttk.Label(eval_frame, textvariable=self.eval_var, font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Right side - Controls
        right_frame = ttk.Frame(main_frame, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_frame.pack_propagate(False)
        
        # Game controls
        game_frame = ttk.LabelFrame(right_frame, text="Game Controls", padding=10)
        game_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(game_frame, text="New Game (White)", command=lambda: self.new_game(chess.WHITE)).pack(fill=tk.X, pady=2)
        ttk.Button(game_frame, text="New Game (Black)", command=lambda: self.new_game(chess.BLACK)).pack(fill=tk.X, pady=2)
        ttk.Button(game_frame, text="Undo Move", command=self.undo_move).pack(fill=tk.X, pady=2)
        ttk.Button(game_frame, text="Engine Move", command=self.request_engine_move).pack(fill=tk.X, pady=2)
        
        # Engine settings
        settings_frame = ttk.LabelFrame(right_frame, text="Engine Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Simulations:").pack(anchor=tk.W)
        self.sim_var = tk.StringVar(value="200")
        sim_spin = ttk.Spinbox(settings_frame, from_=50, to=1000, increment=50, textvariable=self.sim_var, width=10)
        sim_spin.pack(fill=tk.X, pady=2)
        
        ttk.Button(settings_frame, text="Apply Settings", command=self.apply_settings).pack(fill=tk.X, pady=5)
        
        # Training controls
        train_frame = ttk.LabelFrame(right_frame, text="Training", padding=10)
        train_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(train_frame, text="Games per iteration:").pack(anchor=tk.W)
        self.train_games_var = tk.StringVar(value="20")
        ttk.Spinbox(train_frame, from_=5, to=200, increment=5, textvariable=self.train_games_var, width=10).pack(fill=tk.X, pady=2)
        
        ttk.Label(train_frame, text="Iterations:").pack(anchor=tk.W)
        self.train_iters_var = tk.StringVar(value="5")
        ttk.Spinbox(train_frame, from_=1, to=100, increment=1, textvariable=self.train_iters_var, width=10).pack(fill=tk.X, pady=2)
        
        ttk.Label(train_frame, text="Simulations:").pack(anchor=tk.W)
        self.train_sims_var = tk.StringVar(value="100")
        ttk.Spinbox(train_frame, from_=30, to=500, increment=10, textvariable=self.train_sims_var, width=10).pack(fill=tk.X, pady=2)
        
        self.train_btn = ttk.Button(train_frame, text="Start Training", command=self.toggle_training)
        self.train_btn.pack(fill=tk.X, pady=5)
        
        # Training progress
        self.train_progress_var = tk.StringVar(value="Ready")
        ttk.Label(train_frame, textvariable=self.train_progress_var, wraplength=200).pack(fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(train_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Move history
        history_frame = ttk.LabelFrame(right_frame, text="Move History", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.history_text = tk.Text(history_frame, height=10, width=25, font=('Consolas', 9))
        self.history_text.pack(fill=tk.BOTH, expand=True)
        
        # Style
        style = ttk.Style()
        style.configure('TButton', padding=5)
        style.configure('TLabelframe.Label', font=('Segoe UI', 10, 'bold'))
    
    def init_engine(self):
        """Initialize the chess engine."""
        model_path = os.path.join(os.path.dirname(__file__), "models", "chess_model.pt")
        if os.path.exists(model_path):
            self.status_var.set("Loading model...")
            self.engine = ChessEngine(model_path=model_path, num_simulations=200)
            self.status_var.set("Model loaded! Your turn (White)")
        else:
            self.status_var.set("No model found - using random network")
            self.engine = ChessEngine(model_path=None, num_simulations=200)
        
        self.draw_board()
    
    def draw_board(self):
        """Draw the chess board."""
        self.canvas.delete("all")
        board = self.engine.board.board
        
        # Determine orientation
        flip = self.player_color == chess.BLACK
        
        for row in range(8):
            for col in range(8):
                # Calculate display position
                display_row = 7 - row if not flip else row
                display_col = col if not flip else 7 - col
                
                x1 = display_col * self.square_size
                y1 = display_row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Square color
                square = chess.square(col, row)
                is_light = (row + col) % 2 == 1
                
                if square == self.selected_square:
                    color = self.SELECTED
                elif self.selected_square is not None:
                    # Highlight legal move destinations
                    selected_piece = board.piece_at(self.selected_square)
                    if selected_piece:
                        move = chess.Move(self.selected_square, square)
                        # Check for promotion
                        if selected_piece.piece_type == chess.PAWN:
                            if chess.square_rank(square) in [0, 7]:
                                move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
                        if move in board.legal_moves:
                            color = self.HIGHLIGHT
                        else:
                            color = self.LIGHT_SQUARE if is_light else self.DARK_SQUARE
                    else:
                        color = self.LIGHT_SQUARE if is_light else self.DARK_SQUARE
                else:
                    color = self.LIGHT_SQUARE if is_light else self.DARK_SQUARE
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
                
                # Draw piece
                piece = board.piece_at(square)
                if piece:
                    symbol = self.PIECES.get(piece.symbol(), '')
                    self.canvas.create_text(
                        x1 + self.square_size // 2,
                        y1 + self.square_size // 2,
                        text=symbol,
                        font=('Segoe UI Symbol', 40),
                        fill='#1a1a1a' if piece.color == chess.WHITE else '#000000'
                    )
        
        # Draw coordinates
        for i in range(8):
            # Files (a-h)
            file_label = chr(ord('a') + i) if not flip else chr(ord('h') - i)
            self.canvas.create_text(
                i * self.square_size + self.square_size // 2,
                8 * self.square_size - 5,
                text=file_label, font=('Segoe UI', 8), fill='#666', anchor=tk.S
            )
            # Ranks (1-8)
            rank_label = str(8 - i) if not flip else str(i + 1)
            self.canvas.create_text(
                5, i * self.square_size + self.square_size // 2,
                text=rank_label, font=('Segoe UI', 8), fill='#666', anchor=tk.W
            )
    
    def on_square_click(self, event):
        """Handle square click."""
        if self.engine_thinking or self.engine.is_game_over():
            return
        
        # Don't allow moves when it's not player's turn
        if self.engine.board.turn != self.player_color:
            return
        
        flip = self.player_color == chess.BLACK
        
        col = event.x // self.square_size
        row = event.y // self.square_size
        
        if flip:
            col = 7 - col
            row = row
        else:
            row = 7 - row
        
        clicked_square = chess.square(col, row)
        board = self.engine.board.board
        
        if self.selected_square is None:
            # Select a piece
            piece = board.piece_at(clicked_square)
            if piece and piece.color == self.player_color:
                self.selected_square = clicked_square
        else:
            # Try to make a move
            piece = board.piece_at(self.selected_square)
            move = chess.Move(self.selected_square, clicked_square)
            
            # Check for pawn promotion
            if piece and piece.piece_type == chess.PAWN:
                if chess.square_rank(clicked_square) in [0, 7]:
                    move = chess.Move(self.selected_square, clicked_square, promotion=chess.QUEEN)
            
            if move in board.legal_moves:
                self.make_player_move(move)
            else:
                # Maybe selecting a different piece
                new_piece = board.piece_at(clicked_square)
                if new_piece and new_piece.color == self.player_color:
                    self.selected_square = clicked_square
                else:
                    self.selected_square = None
        
        self.draw_board()
    
    def make_player_move(self, move):
        """Make a player move and trigger engine response."""
        self.engine.make_move(move)
        self.selected_square = None
        self.update_history()
        self.draw_board()
        
        if self.engine.is_game_over():
            self.show_game_over()
        else:
            self.status_var.set("Engine thinking...")
            self.root.after(100, self.engine_move)
    
    def engine_move(self):
        """Make engine move in a thread."""
        if self.engine_thinking:
            return
        
        self.engine_thinking = True
        
        def think():
            try:
                best_move, win_prob = self.engine.get_best_move()
                self.root.after(0, lambda: self.complete_engine_move(best_move, win_prob))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))
                self.engine_thinking = False
        
        thread = threading.Thread(target=think, daemon=True)
        thread.start()
    
    def complete_engine_move(self, move, win_prob):
        """Complete engine move on main thread."""
        self.engine.make_move(move)
        self.engine_thinking = False
        
        # Update eval (from engine's perspective)
        eval_pct = win_prob * 100
        self.eval_var.set(f"{eval_pct:.1f}%")
        
        self.update_history()
        self.draw_board()
        
        if self.engine.is_game_over():
            self.show_game_over()
        else:
            turn = "White" if self.engine.board.turn == chess.WHITE else "Black"
            self.status_var.set(f"Your turn ({turn})")
    
    def request_engine_move(self):
        """Request engine to make a move (for engine vs engine or hints)."""
        if not self.engine_thinking and not self.engine.is_game_over():
            self.status_var.set("Engine thinking...")
            self.root.after(100, self.engine_move)
    
    def update_history(self):
        """Update move history display."""
        self.history_text.delete(1.0, tk.END)
        board = self.engine.board.board
        
        moves = list(board.move_stack)
        temp_board = chess.Board()
        
        text = ""
        for i, move in enumerate(moves):
            if i % 2 == 0:
                text += f"{i//2 + 1}. "
            text += temp_board.san(move) + " "
            temp_board.push(move)
            if i % 2 == 1:
                text += "\n"
        
        self.history_text.insert(1.0, text)
    
    def new_game(self, player_color):
        """Start a new game."""
        self.player_color = player_color
        self.engine.new_game()
        self.selected_square = None
        self.history_text.delete(1.0, tk.END)
        self.eval_var.set("50.0%")
        self.draw_board()
        
        if player_color == chess.BLACK:
            self.status_var.set("Engine thinking...")
            self.root.after(100, self.engine_move)
        else:
            self.status_var.set("Your turn (White)")
    
    def undo_move(self):
        """Undo last two moves (player + engine)."""
        board = self.engine.board.board
        if len(board.move_stack) >= 2:
            board.pop()
            board.pop()
            self.selected_square = None
            self.update_history()
            self.draw_board()
            self.status_var.set("Moves undone. Your turn.")
    
    def flip_board(self):
        """Flip the board orientation."""
        self.player_color = not self.player_color
        self.draw_board()
    
    def apply_settings(self):
        """Apply engine settings."""
        try:
            sims = int(self.sim_var.get())
            self.engine.mcts.num_simulations = sims
            messagebox.showinfo("Settings", f"Simulations set to {sims}")
        except ValueError:
            messagebox.showerror("Error", "Invalid simulation count")
    
    def load_model(self):
        """Load a model file."""
        filepath = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")],
            initialdir=os.path.join(os.path.dirname(__file__), "models")
        )
        if filepath:
            try:
                self.engine = ChessEngine(model_path=filepath, num_simulations=int(self.sim_var.get()))
                self.new_game(self.player_color)
                messagebox.showinfo("Success", f"Model loaded from {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def save_model(self):
        """Save current model."""
        filepath = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pt",
            filetypes=[("PyTorch Model", "*.pt")],
            initialdir=os.path.join(os.path.dirname(__file__), "models")
        )
        if filepath:
            try:
                self.engine.neural_network.save(filepath)
                messagebox.showinfo("Success", f"Model saved to {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {e}")
    
    def show_game_over(self):
        """Show game over dialog."""
        result = self.engine.get_result()
        if result == '1-0':
            msg = "White wins!"
        elif result == '0-1':
            msg = "Black wins!"
        else:
            msg = "Draw!"
        
        self.status_var.set(f"Game Over: {msg}")
        messagebox.showinfo("Game Over", msg)
    
    def toggle_training(self):
        """Start or stop training."""
        if self.training_active:
            self.training_active = False
            self.train_btn.config(text="Start Training")
            self.train_progress_var.set("Training stopped")
        else:
            self.training_active = True
            self.train_btn.config(text="Stop Training")
            self.start_training()
    
    def start_training(self):
        """Start training in a background thread."""
        try:
            num_games = int(self.train_games_var.get())
            num_iters = int(self.train_iters_var.get())
            num_sims = int(self.train_sims_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid training parameters")
            self.training_active = False
            self.train_btn.config(text="Start Training")
            return
        
        def train_thread():
            try:
                nn = ChessNeuralNetwork()
                trainer = SelfPlayTrainer(nn, num_simulations=num_sims)
                
                total_games = num_games * num_iters
                games_done = 0
                
                for iteration in range(1, num_iters + 1):
                    if not self.training_active:
                        break
                    
                    self.root.after(0, lambda i=iteration: self.train_progress_var.set(
                        f"Iteration {i}/{num_iters}..."
                    ))
                    
                    # Generate games
                    for game_num in range(num_games):
                        if not self.training_active:
                            break
                        
                        trainer.self_play_game(temperature=1.0, temp_threshold=30, game_num=game_num+1)
                        games_done += 1
                        
                        progress = (games_done / total_games) * 100
                        self.root.after(0, lambda p=progress, g=games_done: self.update_training_progress(p, g, total_games))
                    
                    if not self.training_active:
                        break
                    
                    # Train
                    self.root.after(0, lambda: self.train_progress_var.set("Training neural network..."))
                    states, policies, values = trainer.replay_buffer.get_all()
                    if len(states) > 0:
                        nn.train(states, policies, values, epochs=5, batch_size=64)
                    
                    # Save checkpoint
                    checkpoint_path = os.path.join(os.path.dirname(__file__), "models", f"chess_model_iter{iteration}.pt")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    nn.save(checkpoint_path)
                
                # Save final model
                if self.training_active:
                    final_path = os.path.join(os.path.dirname(__file__), "models", "chess_model.pt")
                    nn.save(final_path)
                    self.root.after(0, lambda: self.training_complete(final_path))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            finally:
                self.root.after(0, self.training_finished)
        
        self.training_thread = threading.Thread(target=train_thread, daemon=True)
        self.training_thread.start()
    
    def update_training_progress(self, progress, games_done, total_games):
        """Update training progress display."""
        self.progress_bar['value'] = progress
        self.train_progress_var.set(f"Game {games_done}/{total_games}")
    
    def training_complete(self, model_path):
        """Called when training completes."""
        self.train_progress_var.set("Training complete!")
        messagebox.showinfo("Training Complete", f"Model saved to {model_path}\n\nReload the model to play against it.")
    
    def training_finished(self):
        """Reset training UI state."""
        self.training_active = False
        self.train_btn.config(text="Start Training")
        self.progress_bar['value'] = 0


def main():
    """Main entry point."""
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
