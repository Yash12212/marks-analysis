import pygame
import chess
import chess.engine

# --- Constants ---
BOARD_SIZE = 8
SQUARE_SIZE = 60
WIDTH = BOARD_SIZE * SQUARE_SIZE
HEIGHT = BOARD_SIZE * SQUARE_SIZE
LIGHT_SQUARE_COLOR = (240, 217, 181)
DARK_SQUARE_COLOR = (181, 136, 99)
HIGHLIGHT_COLOR_SELECTED = (255, 255, 102, 150)
HIGHLIGHT_COLOR_MOVE = (173, 255, 47, 150)
PIECE_FONT_SIZE = 45
GAME_OVER_FONT_SIZE = 40
INFO_FONT_SIZE = 20

# --- Initialize Pygame ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess (Smaller UI, Optimized AI)")

# --- Initialize Font ---
font_path = pygame.font.match_font("Chess Alpha")
piece_font = pygame.font.Font(font_path, PIECE_FONT_SIZE)
game_over_font = pygame.font.Font(None, GAME_OVER_FONT_SIZE)
info_font = pygame.font.Font(None, INFO_FONT_SIZE)

# --- Chess Piece Symbols ---
piece_symbols = {
    chess.PAWN:   'P',
    chess.ROOK:   'R',
    chess.KNIGHT: 'H',
    chess.BISHOP: 'B',
    chess.QUEEN:  'Q',
    chess.KING:   'K',
    chess.PAWN | chess.BLACK:   'p',
    chess.ROOK | chess.BLACK:   'r',
    chess.KNIGHT | chess.BLACK: 'h',
    chess.BISHOP | chess.BLACK: 'b',
    chess.QUEEN | chess.BLACK:  'q',
    chess.KING | chess.BLACK:   'k'
}

# --- Chess Board Setup ---
board = chess.Board()
selected_square = None
possible_moves = []

# --- Optimized Minimax AI Functions ---
def get_best_move_minimax_ab(board, depth):
    """Uses Minimax with Alpha-Beta pruning to find the best move."""
    if board.is_game_over():
        return None

    best_move = None
    best_eval = -float('inf')
    alpha = -float('inf')
    beta = float('inf')

    # Move ordering: Captures first (simple heuristic)
    ordered_moves = sorted(board.legal_moves, key=lambda move: board.is_capture(move), reverse=True)

    for move in ordered_moves:
        board.push(move)
        evaluation = minimax_ab(board, depth - 1, False, alpha, beta)
        board.pop()
        if evaluation > best_eval:
            best_eval = evaluation
            best_move = move
        alpha = max(alpha, best_eval) # Update alpha
    return best_move

def minimax_ab(board, depth, maximizing_player, alpha, beta):
    """Recursive Minimax with Alpha-Beta pruning."""
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    if maximizing_player:
        max_eval = -float('inf')
        # Move ordering: Captures first (simple heuristic)
        ordered_moves = sorted(board.legal_moves, key=lambda move: board.is_capture(move), reverse=True)
        for move in ordered_moves:
            board.push(move)
            evaluation = minimax_ab(board, depth - 1, False, alpha, beta)
            board.pop()
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, max_eval) # Update alpha
            if beta <= alpha:
                break # Beta cutoff
        return max_eval
    else:
        min_eval = float('inf')
        # Move ordering: Captures first (simple heuristic)
        ordered_moves = sorted(board.legal_moves, key=lambda move: board.is_capture(move), reverse=True)
        for move in ordered_moves:
            board.push(move)
            evaluation = minimax_ab(board, depth - 1, True, alpha, beta)
            board.pop()
            min_eval = min(min_eval, evaluation)
            beta = min(beta, min_eval) # Update beta
            if beta <= alpha:
                break # Alpha cutoff
        return min_eval

def evaluate_board(board):
    """Basic evaluation function (material count)."""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    evaluation = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            evaluation += value if piece.color == chess.WHITE else -value
    return evaluation

# --- Drawing Functions --- (No changes needed in drawing functions)
def draw_board():
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            color = LIGHT_SQUARE_COLOR if (row + col) % 2 == 0 else DARK_SQUARE_COLOR
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                piece_symbol_key = piece.piece_type
                piece_color = (255, 255, 255)
            else:
                piece_symbol_key = piece.piece_type | chess.BLACK
                piece_color = (0, 0, 0)

            piece_symbol = piece_symbols[piece_symbol_key]
            text_surface = piece_font.render(piece_symbol, True, piece_color)
            text_rect = text_surface.get_rect(center=pygame.Rect(
                chess.square_file(square) * SQUARE_SIZE,
                (7 - chess.square_rank(square)) * SQUARE_SIZE,
                SQUARE_SIZE,
                SQUARE_SIZE).center)
            screen.blit(text_surface, text_rect)

def highlight_square(square):
    if square is not None:
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        square_rect = pygame.Rect(file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        surface = pygame.Surface(square_rect.size, pygame.SRCALPHA)
        surface.fill(HIGHLIGHT_COLOR_SELECTED)
        screen.blit(surface, square_rect.topleft)

def highlight_moves(moves):
    for move in moves:
        to_square = move.to_square
        rank = chess.square_rank(to_square)
        file = chess.square_file(to_square)
        square_rect = pygame.Rect(file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        surface = pygame.Surface(square_rect.size, pygame.SRCALPHA)
        surface.fill(HIGHLIGHT_COLOR_MOVE)
        screen.blit(surface, square_rect.topleft)

def draw_game_over_screen(text):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    screen.blit(overlay, (0, 0))

    text_surface = game_over_font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))

    background_rect = text_rect.inflate(20, 20)
    pygame.draw.rect(screen, (50, 50, 50), background_rect)
    pygame.draw.rect(screen, (200, 200, 200), background_rect, 2)

    screen.blit(text_surface, text_rect)

def draw_turn_info(turn):
    turn_text = "Your Turn" if turn == chess.WHITE else "AI Thinking..."
    text_surface = info_font.render(turn_text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(topleft=(10, 10))
    screen.blit(text_surface, text_rect)


# --- Game Loop Setup ---
board = chess.Board()
selected_square = None
possible_moves = []

running = True
player_turn = True
ai_depth = 4

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if player_turn and not board.is_game_over():
                click_pos = pygame.mouse.get_pos()
                clicked_square = chess.square(click_pos[0] // SQUARE_SIZE, 7 - (click_pos[1] // SQUARE_SIZE))
                if selected_square is None:
                    if board.piece_at(clicked_square) is not None and board.piece_at(clicked_square).color == board.turn:
                        selected_square = clicked_square
                        possible_moves = [move for move in board.legal_moves if move.from_square == selected_square]
                else:
                    move = chess.Move(selected_square, clicked_square)
                    if move in board.legal_moves:
                        board.push(move)
                        selected_square = None
                        possible_moves = []
                        player_turn = False
                    else:
                        if board.piece_at(clicked_square) is not None and board.piece_at(clicked_square).color == board.turn:
                            selected_square = clicked_square
                            possible_moves = [move for move in board.legal_moves if move.from_square == selected_square]
                        else:
                            selected_square = None
                            possible_moves = []

    # --- AI Move ---
    if not player_turn and not board.is_game_over():
        ai_move = get_best_move_minimax_ab(board, ai_depth) # Use optimized AI function
        if ai_move:
            board.push(ai_move)
        player_turn = True

    # --- Drawing ---
    draw_board()
    highlight_moves(possible_moves)
    highlight_square(selected_square)
    draw_pieces(board)
    draw_turn_info(board.turn)

    if board.is_game_over():
        game_over_text = ""
        if board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "Black"
            game_over_text = f"Checkmate! {winner} wins!"
        elif board.is_stalemate():
            game_over_text = "Stalemate!"
        elif board.is_insufficient_material():
            game_over_text = "Draw: Insufficient material!"
        elif board.is_seventyfive_moves():
            game_over_text = "Draw: 75-moves rule!"
        elif board.is_fivefold_repetition():
            game_over_text = "Draw: Fivefold repetition!"
        elif board.is_variant_end():
            game_over_text = "Game variant end!"
        elif board.is_game_over(claim_draw=True):
            game_over_text = "Draw!"

        if game_over_text:
            draw_game_over_screen(game_over_text)

    pygame.display.flip()

pygame.quit()