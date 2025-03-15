import pygame
import chess
import time
import threading
from typing import Optional, List, Tuple

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
MAX_AI_DEPTH = 10
NULL_MOVE_REDUCTION = 2
DEVELOPMENT_BONUS = 15
BISHOP_PAIR_BONUS = 30
PASSED_PAWN_BONUS_FACTOR = 10
ISOLATED_PAWN_PENALTY = 20
SPACE_BONUS_FACTOR = 0.10
KING_SAFETY_WEIGHT_FACTOR = 1.2
MATERIAL_WEIGHT = 1.5  # Increased material weight

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess (Improved Search & Defensive Play)")
clock = pygame.time.Clock()

font_path = pygame.font.match_font("Chess Alpha")
piece_font = pygame.font.Font(font_path, PIECE_FONT_SIZE) if font_path else pygame.font.Font(None, PIECE_FONT_SIZE)
game_over_font = pygame.font.Font(None, GAME_OVER_FONT_SIZE)
info_font = pygame.font.Font(None, INFO_FONT_SIZE)

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

pawn_table = [
     0,   0,   0,   0,   0,   0,   0,   0,
    50,  50,  50,  50,  50,  50,  50,  50,
    10,  10,  20,  25,  25,  20,  10,  10,
     5,   5,  15,  20,  20,  15,   5,   5,
     0,   0,   0,  15,  15,   0,   0,   0,
     0,   0,  -5,   0,   0,  -5,   0,   0,
     0,   5,   5, -10, -10,   5,   5,   0,
     0,   0,   0,   0,   0,   0,   0,   0
]

knight_table = [
   -50, -40, -30, -30, -30, -30, -40, -50,
   -40, -20,   0,   5,   5,   0, -20, -40,
   -30,   5,  15,  20,  20,  15,   5, -30,
   -30,   0,  20,  25,  25,  20,   0, -30,
   -30,   5,  15,  20,  20,  15,   5, -30,
   -30,   0,  10,  15,  15,  10,   0, -30,
   -40, -20,   0,   5,   5,   0, -20, -40,
   -50, -40, -30, -30, -30, -30, -40, -50
]

bishop_table = [
   -20, -10, -10, -10, -10, -10, -10, -20,
   -10,   5,   5,   5,   5,   5,   5, -10,
   -10,   5,  10,  10,  10,  10,   5, -10,
   -10,   0,  10,  15,  15,  10,   0, -10,
   -10,   5,  10,  10,  10,  10,   5, -10,
   -10,   0,   5,  10,  10,   5,   0, -10,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -20, -10, -10, -10, -10, -10, -10, -20
]

rook_table = [
     0,   0,   0,   5,   5,   0,   0,   0,
   -5,   0,   0,   0,   0,   0,   0,  -5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
   -5,   0,   0,   0,   0,   0,   0,  -5,
    5,  10,  10,  10,  10,  10,  10,   5,
     0,   0,   0,   0,   0,   0,   0,   0
]

queen_table = [
   -20, -10, -10,  -5,  -5, -10, -10, -20,
   -10,   0,   5,   0,   0,   0,   0, -10,
   -10,   5,   5,   5,   5,   5,   0, -10,
    -5,   0,   5,   5,   5,   5,   0,  -5,
     0,   0,   5,   5,   5,   5,   0,  -5,
   -10,   0,   5,   5,   5,   5,   0, -10,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -20, -10, -10,  -5,  -5, -10, -10, -20
]

king_table = [
   -100, -120, -120, -140, -140, -120, -120, -100,
   -100, -120, -120, -140, -140, -120, -120, -100,
   -100, -120, -120, -140, -140, -120, -120, -100,
   -100, -120, -120, -140, -140, -120, -120, -100,
   -80,  -100, -100, -120, -120, -100, -100, -80,
   -60,  -80,  -80,  -80,  -80,  -80,  -80,  -60,
    20,  20,   0,   0,   0,   0,  20,  20,
    30,  40,  20,   0,   0,  20,  40,  30
]

central_squares = []
white_dev_squares = [chess.A1, chess.B1, chess.C1, chess.D1, chess.E1, chess.F1, chess.G1, chess.H1,
                     chess.A2, chess.B2, chess.C2, chess.D2, chess.E2, chess.F2, chess.G2, chess.H2]
black_dev_squares = [chess.A8, chess.B8, chess.C8, chess.D8, chess.E8, chess.F8, chess.G8, chess.H8,
                     chess.A7, chess.B7, chess.C7, chess.D7, chess.E7, chess.F7, chess.G7, chess.H7]


class ChessGame:
    def __init__(self) -> None:
        self.board: chess.Board = chess.Board()
        self.selected_square: Optional[int] = None
        self.possible_moves: List[chess.Move] = []
        self.player_turn: bool = True
        self.transposition_table: dict[str, dict] = {}
        self.killer_moves: dict[int, List[chess.Move]] = {}
        self.history_heuristic: dict[Tuple[chess.PieceType, chess.Square, chess.Square], int] = {}
        self.time_limit: float = 5.0
        self.pondering_move: Optional[chess.Move] = None
        self.pondering_thread = None
        self.material_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000,
        }

    def is_endgame(self) -> bool:
        non_king_material = 0
        for _, piece in self.board.piece_map().items():
            if piece.piece_type != chess.KING:
                non_king_material += self.material_values[piece.piece_type]
        return non_king_material < 1400

    def get_attacker_value(self, square: chess.Square, color: chess.Color) -> float:
        value = 0
        opponent_color = not color
        for attacker_square in self.board.attackers(opponent_color, square):
            attacker_piece = self.board.piece_at(attacker_square)
            if attacker_piece:
                value += self.material_values.get(attacker_piece.piece_type, 0)
        return value

    def get_defender_value(self, square: chess.Square, color: chess.Color) -> float:
        value = 0
        for defender_square in self.board.attackers(color, square):
            defender_piece = self.board.piece_at(defender_square)
            if defender_piece:
                value += self.material_values.get(defender_piece.piece_type, 0)
        return value

    def evaluate_board(self) -> float:
        score = 0.0
        white_bishops = 0
        black_bishops = 0
        development_bonus_white = 0
        development_bonus_black = 0
        space_white = 0
        space_black = 0
        pawn_chain_bonus = 5
        backward_pawn_penalty = 15

        for square, piece in self.board.piece_map().items():
            piece_value = self.material_values[piece.piece_type]
            bonus = 0

            if piece.piece_type == chess.PAWN:
                bonus += pawn_table[square] if piece.color == chess.WHITE else pawn_table[chess.square_mirror(square)]
                if self.is_isolated_pawn(square, piece.color):
                    bonus -= ISOLATED_PAWN_PENALTY
                if self.is_passed_pawn(square, piece.color):
                    bonus += PASSED_PAWN_BONUS_FACTOR * chess.square_rank(square) if piece.color == chess.WHITE else PASSED_PAWN_BONUS_FACTOR * (7 - chess.square_rank(square))
                if self.is_pawn_in_chain(square, piece.color):
                    bonus += pawn_chain_bonus
                if self.is_backward_pawn(square, piece.color):
                    bonus -= backward_pawn_penalty
            elif piece.piece_type == chess.KNIGHT:
                bonus += knight_table[square] if piece.color == chess.WHITE else knight_table[chess.square_mirror(square)]
                if piece.color == chess.WHITE and square not in white_dev_squares:
                    development_bonus_white += DEVELOPMENT_BONUS
                elif piece.color == chess.BLACK and square not in black_dev_squares:
                    development_bonus_black += DEVELOPMENT_BONUS
            elif piece.piece_type == chess.BISHOP:
                bonus += bishop_table[square] if piece.color == chess.WHITE else bishop_table[chess.square_mirror(square)]
                if piece.color == chess.WHITE:
                    white_bishops += 1
                    if square not in white_dev_squares:
                        development_bonus_white += DEVELOPMENT_BONUS
                else:
                    black_bishops += 1
                    if square not in black_dev_squares:
                        development_bonus_black += DEVELOPMENT_BONUS
            elif piece.piece_type == chess.ROOK:
                bonus += rook_table[square] if piece.color == chess.WHITE else rook_table[chess.square_mirror(square)]
            elif piece.piece_type == chess.QUEEN:
                bonus += queen_table[square] if piece.color == chess.WHITE else queen_table[chess.square_mirror(square)]
            elif piece.piece_type == chess.KING:
                bonus += king_table[square] if piece.color == chess.WHITE else king_table[chess.square_mirror(square)]

            attackers_value = self.get_attacker_value(square, piece.color)
            defenders_value = self.get_defender_value(square, piece.color)
            if attackers_value > defenders_value:
                diff = attackers_value - defenders_value
                if defenders_value == 0:
                    bonus -= piece_value * (diff / piece_value) * 1.5
                else:
                    bonus -= piece_value * (diff / (piece_value * 2))

            rank = chess.square_rank(square)
            if piece.color == chess.WHITE and rank >= 4:
                space_white += 1
            elif piece.color == chess.BLACK and rank <= 3:
                space_black += 1

            if piece.color == chess.WHITE:
                score += piece_value * MATERIAL_WEIGHT + bonus # Material weight applied here
            else:
                score -= piece_value * MATERIAL_WEIGHT + bonus # Material weight applied here

        if white_bishops >= 2:
            score += BISHOP_PAIR_BONUS
        if black_bishops >= 2:
            score -= BISHOP_PAIR_BONUS

        current_turn = self.board.turn
        self.board.turn = chess.WHITE
        mobility_white = self.board.legal_moves.count()
        self.board.turn = chess.BLACK
        mobility_black = self.board.legal_moves.count()
        self.board.turn = current_turn
        score += (mobility_white - mobility_black) * 6

        score += development_bonus_white - development_bonus_black

        score += (space_white - space_black) * SPACE_BONUS_FACTOR

        white_king_square = self.board.king(chess.WHITE)
        black_king_square = self.board.king(chess.BLACK)
        if white_king_square:
            score += self.king_safety_bonus(white_king_square, chess.WHITE) * KING_SAFETY_WEIGHT_FACTOR
        if black_king_square:
            score -= self.king_safety_bonus(black_king_square, chess.BLACK) * KING_SAFETY_WEIGHT_FACTOR

        return score

    def is_isolated_pawn(self, square: int, color: bool) -> bool:
        file = chess.square_file(square)
        for adj_file in [file - 1, file + 1]:
            if 0 <= adj_file < 8:
                for rank in range(8):
                    adj_square = chess.square(adj_file, rank)
                    piece = self.board.piece_at(adj_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        return False
        return True

    def is_passed_pawn(self, square: int, color: bool) -> bool:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        direction = 1 if color == chess.WHITE else -1
        for f in range(max(0, file - 1), min(8, file + 2)):
            for r in range(rank + direction, 8 if color == chess.WHITE else -1, direction):
                sq = chess.square(f, r)
                piece = self.board.piece_at(sq)
                if piece and piece.color != color and piece.piece_type == chess.PAWN:
                    return False
        return True

    def king_safety_bonus(self, king_square: int, color: bool) -> float:
        bonus = 0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        attackers_count = 0
        pawn_shield_count = 0
        close_enemy_pieces = 0
        heavy_piece_attackers = 0
        distance_to_king_penalty = 0

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                sq_file = king_file + dx
                sq_rank = king_rank + dy
                if 0 <= sq_file < 8 and 0 <= sq_rank < 8:
                    sq = chess.square(sq_file, sq_rank)
                    piece = self.board.piece_at(sq)
                    if piece and piece.color != color:
                        attackers_count += 1
                        distance = max(abs(dx), abs(dy))
                        distance_to_king_penalty += (4 - distance) * 2
                        if piece.piece_type not in [chess.PAWN, chess.KING]:
                            close_enemy_pieces += 1
                            if piece.piece_type in [chess.ROOK, chess.QUEEN, chess.KNIGHT, chess.BISHOP]:
                                heavy_piece_attackers += 1
                    if piece and piece.color == color and piece.piece_type == chess.PAWN:
                        pawn_shield_count += 1

        bonus -= attackers_count * 7
        bonus -= close_enemy_pieces * 5
        bonus -= heavy_piece_attackers * 8
        bonus -= distance_to_king_penalty
        bonus += pawn_shield_count * 3

        if not self.is_endgame():
            if 2 <= king_file <= 5:
                file_open = True
                for rank in range(8):
                    sq = chess.square(king_file, rank)
                    if self.board.piece_at(sq) and self.board.piece_at(sq).piece_type == chess.PAWN:
                        file_open = False
                        break
                if file_open:
                    bonus -= 25

        if not self.is_endgame():
            if color == chess.WHITE:
                if self.board.has_kingside_castling_rights(chess.WHITE) or self.board.has_queenside_castling_rights(chess.WHITE):
                    bonus += 15
            else:
                if self.board.has_kingside_castling_rights(chess.BLACK) or self.board.has_queenside_castling_rights(chess.BLACK):
                    bonus += 15

        return bonus

    def gives_check(self, move: chess.Move) -> bool:
        if not self.board.is_pseudo_legal(move):
            return False
        self.board.push(move)
        check_given = self.board.is_check()
        self.board.pop()
        return check_given

    def static_exchange_evaluation(self, move: chess.Move) -> float:
        if not self.board.is_capture(move):
            return 0
        attacker = self.board.piece_at(move.from_square)
        captured = self.board.piece_at(move.to_square)
        if captured is None:
            captured_piece_type = chess.PAWN
        else:
            captured_piece_type = captured.piece_type

        if attacker is None or captured_piece_type is None:
            return 0

        if self.material_values[attacker.piece_type] >= self.material_values[captured_piece_type]:
            return 1
        else:
            return -1

    is_pawn_in_chain = evaluate_board.is_pawn_in_chain = lambda self, square, color: False
    is_backward_pawn = evaluate_board.is_backward_pawn = lambda self, square, color: False

    def move_ordering_key(self, move: chess.Move, depth: int) -> int:
        score = 0

        if self.board.is_capture(move):
            captured_piece = self.board.piece_at(move.to_square)
            attacker_piece = self.board.piece_at(move.from_square)
            if captured_piece and attacker_piece:
                mvv_lva_score = 10 * self.material_values.get(captured_piece.piece_type, 0) - self.material_values.get(attacker_piece.piece_type, 0)
                score += mvv_lva_score + 10000

            see_value = self.static_exchange_evaluation(move)
            if see_value > 0:
                score += 1000
            else:
                score -= 500

        if self.gives_check(move):
            score += 200
        if depth in self.killer_moves and move in self.killer_moves[depth]:
            score += 800
        piece_type = self.board.piece_type_at(move.from_square)
        history_value = self.history_heuristic.get((piece_type, move.from_square, move.to_square), 0)
        score += history_value // 4
        fen = self.board.fen()
        if fen in self.transposition_table:
            entry = self.transposition_table[fen]
            if 'best_move' in entry and entry['best_move'] == move:
                score += 15000

        if not self.board.is_capture(move) and not self.gives_check(move):
            king_square = self.board.king(self.board.turn)
            if king_square:
                original_king_safety = self.king_safety_bonus(king_square, self.board.turn)
                self.board.push(move)
                new_king_square = self.board.king(self.board.turn)
                new_king_safety = self.king_safety_bonus(new_king_square, self.board.turn)
                self.board.pop()
                safety_improvement = new_king_safety - original_king_safety
                score += safety_improvement * 5

        return score

    def quiescence(self, alpha: float, beta: float, deadline: float) -> float:
        stand_pat = self.evaluate_board()
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        capture_moves = [m for m in self.board.legal_moves if self.board.is_capture(m)]
        capture_moves.sort(key=lambda m: self.move_ordering_key(m, 0), reverse=True)

        for move in capture_moves:
            if time.time() > deadline:
                break
            self.board.push(move)
            score = -self.quiescence(-beta, -alpha, deadline)
            self.board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def minimax_ab(self, depth: int, maximizing_player: bool, alpha: float, beta: float, deadline: float, is_pv_node: bool) -> float:
        if time.time() > deadline:
            return self.evaluate_board()

        key = self.board.fen()
        tt_entry = self.transposition_table.get(key)
        if tt_entry and tt_entry['depth'] >= depth:
            if tt_entry['flag'] == 'EXACT':
                return tt_entry['score']
            elif tt_entry['flag'] == 'LOWERBOUND':
                alpha = max(alpha, tt_entry['score'])
            elif tt_entry['flag'] == 'UPPERBOUND':
                beta = min(beta, tt_entry['score'])
            if alpha >= beta:
                return tt_entry['score']

        if depth >= 3 and not self.board.is_check() and not self.board.is_game_over() and not self.is_endgame():
            self.board.push(chess.Move.null())
            score = -self.minimax_ab(depth - 1 - NULL_MOVE_REDUCTION, not maximizing_player, -beta, -beta + 1, deadline, False)
            self.board.pop()
            if score >= beta:
                self.transposition_table[key] = {'score': beta, 'depth': depth, 'flag': 'LOWERBOUND'}
                return beta

        if depth == 0 or self.board.is_game_over():
            evaluation = self.quiescence(alpha, beta, deadline)
            self.transposition_table[key] = {'score': evaluation, 'depth': depth, 'flag': 'EXACT'}
            return evaluation

        moves = list(self.board.legal_moves)
        moves.sort(key=lambda move: self.move_ordering_key(move, depth), reverse=True)

        best_move = None
        best_eval = -float('inf') if maximizing_player else float('inf')
        pv_found = False

        if maximizing_player:
            max_eval = -float('inf')
            for move_index, move in enumerate(moves):
                self.board.push(move)
                extension = 1 if self.gives_check(move) and depth >= 2 else 0
                reduction = 0
                if depth >= 3 and move_index >= 2 and not is_pv_node and not self.board.is_capture(move) and not self.gives_check(move):
                    reduction = 1
                    if depth >= 5 and move_index >= 4:
                        reduction = 2

                reduced_depth = depth - 1 + extension - reduction
                reduced_depth = max(0, reduced_depth)

                if is_pv_node and pv_found and move_index > 0:
                    eval_score = -self.minimax_ab(reduced_depth, False, -alpha -1, -alpha, deadline, False)
                    if eval_score > alpha and eval_score < beta:
                        eval_score = -self.minimax_ab(depth - 1 + extension, False, -beta, -alpha, deadline, True)
                else:
                    eval_score = -self.minimax_ab(reduced_depth, False, -beta, -alpha, deadline, is_pv_node)

                self.board.pop()

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                    if is_pv_node:
                        pv_found = True

                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    if not self.board.is_capture(move):
                        self.killer_moves.setdefault(depth, [])
                        if move not in self.killer_moves[depth]:
                            self.killer_moves[depth].append(move)
                            if len(self.killer_moves[depth]) > 2:
                                self.killer_moves[depth].pop(0)
                    piece_type = self.board.piece_type_at(move.from_square)
                    self.history_heuristic[(piece_type, move.from_square, move.to_square)] = self.history_heuristic.get((piece_type, move.from_square, move.to_square), 0) + depth * depth
                    break
            best_eval = max_eval
            flag = 'EXACT'
            if max_eval <= alpha:
                flag = 'UPPERBOUND'
            elif max_eval >= beta:
                flag = 'LOWERBOUND'

        else:
            min_eval = float('inf')
            for move_index, move in enumerate(moves):
                self.board.push(move)
                extension = 1 if self.gives_check(move) and depth >= 2 else 0

                reduction = 0
                if depth >= 3 and move_index >= 2 and not is_pv_node and not self.board.is_capture(move) and not self.gives_check(move):
                    reduction = 1
                    if depth >= 5 and move_index >= 4:
                        reduction = 2

                reduced_depth = depth - 1 + extension - reduction
                reduced_depth = max(0, reduced_depth)

                if is_pv_node and pv_found and move_index > 0:
                    eval_score = self.minimax_ab(reduced_depth, True, alpha, alpha + 1, deadline, False)
                    if eval_score < beta and eval_score > alpha:
                         eval_score = self.minimax_ab(depth - 1 + extension, True, alpha, beta, deadline, True)
                else:
                    eval_score = self.minimax_ab(reduced_depth, True, alpha, beta, deadline, is_pv_node)

                self.board.pop()

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                    if is_pv_node:
                        pv_found = True

                beta = min(beta, min_eval)
                if beta <= alpha:
                    if not self.board.is_capture(move):
                        self.killer_moves.setdefault(depth, [])
                        if move not in self.killer_moves[depth]:
                            self.killer_moves[depth].append(move)
                            if len(self.killer_moves[depth]) > 2:
                                self.killer_moves[depth].pop(0)
                    piece_type = self.board.piece_type_at(move.from_square)
                    self.history_heuristic[(piece_type, move.from_square, move.to_square)] = self.history_heuristic.get((piece_type, move.from_square, move.to_square), 0) + depth * depth
                    break
            best_eval = min_eval
            flag = 'EXACT'
            if min_eval <= alpha:
                flag = 'UPPERBOUND'
            elif min_eval >= beta:
                flag = 'LOWERBOUND'


        self.transposition_table[key] = {'score': best_eval, 'depth': depth, 'flag': flag, 'best_move': best_move}
        return best_eval

    def get_best_move_depth_with_aspiration(
        self, depth: int, guess: float, window: float, deadline: float
    ) -> Tuple[Optional[chess.Move], float]:
        best_move: Optional[chess.Move] = None
        best_eval = -float('inf')
        alpha = guess - window
        beta = guess + window

        best_eval = self.minimax_ab(depth, True, alpha, beta, deadline, True)

        if best_eval <= alpha:
            alpha = -float('inf')
            best_eval = self.minimax_ab(depth, True, alpha, beta, deadline, True)
        elif best_eval >= beta:
            beta = float('inf')
            best_eval = self.minimax_ab(depth, True, alpha, beta, deadline, True)

        key = self.board.fen()
        if key in self.transposition_table and 'best_move' in self.transposition_table[key]:
            best_move = self.transposition_table[key]['best_move']

        return best_move, best_eval

    def get_best_move(self, max_depth: int) -> Optional[chess.Move]:
        best_move: Optional[chess.Move] = None
        guess = 0.0
        window = 50.0
        self.transposition_table.clear()
        self.killer_moves.clear()
        self.history_heuristic.clear()
        start_time = time.time()
        deadline = start_time + self.time_limit

        for depth in range(1, max_depth + 1):
            if time.time() > deadline:
                break
            current_best, best_eval = self.get_best_move_depth_with_aspiration(depth, guess, window, deadline)
            if current_best is not None:
                best_move = current_best
                guess = best_eval
                window = max(20, abs(guess) * 0.05)
            print(f"Depth {depth}: Eval={best_eval:.2f}, Move={best_move}")

        return best_move

    def handle_mouse_click(self, pos: tuple) -> None:
        if self.board.is_game_over():
            return
        col = pos[0] // SQUARE_SIZE
        row = 7 - (pos[1] // SQUARE_SIZE)
        clicked_square = chess.square(col, row)
        if self.selected_square is None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.selected_square = clicked_square
                self.possible_moves = [move for move in self.board.legal_moves if move.from_square == clicked_square]
        else:
            move = chess.Move(self.selected_square, clicked_square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.possible_moves = []
                self.player_turn = False
                self.stop_pondering()
            else:
                piece = self.board.piece_at(clicked_square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = clicked_square
                    self.possible_moves = [move for move in self.board.legal_moves if move.from_square == clicked_square]
                else:
                    self.selected_square = None
                    self.possible_moves = []

    def draw_board(self) -> None:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = LIGHT_SQUARE_COLOR if (row + col) % 2 == 0 else DARK_SQUARE_COLOR
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def draw_pieces(self) -> None:
        for square, piece in self.board.piece_map().items():
            piece_key = piece.piece_type if piece.color == chess.WHITE else piece.piece_type | chess.BLACK
            piece_color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
            piece_symbol = piece_symbols.get(piece_key, '?')
            text_surface = piece_font.render(piece_symbol, True, piece_color)
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            square_rect = pygame.Rect(file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            text_rect = text_surface.get_rect(center=square_rect.center)
            screen.blit(text_surface, text_rect)

    def highlight_square(self, square: Optional[int]) -> None:
        if square is not None:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            square_rect = pygame.Rect(file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            surface = pygame.Surface(square_rect.size, pygame.SRCALPHA)
            surface.fill(HIGHLIGHT_COLOR_SELECTED)
            screen.blit(surface, square_rect.topleft)

    def highlight_moves(self) -> None:
        for move in self.possible_moves:
            to_square = move.to_square
            rank = chess.square_rank(to_square)
            file = chess.square_file(to_square)
            square_rect = pygame.Rect(file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            surface = pygame.Surface(square_rect.size, pygame.SRCALPHA)
            surface.fill(HIGHLIGHT_COLOR_MOVE)
            screen.blit(surface, square_rect.topleft)

    def draw_turn_info(self) -> None:
        turn_text = "Your Turn" if self.board.turn == chess.WHITE else "AI Thinking..."
        if not self.player_turn and self.pondering_move:
            turn_text = "AI Pondering..."
        text_surface = info_font.render(turn_text, True, (0, 0, 0))
        screen.blit(text_surface, (10, 10))

    def draw_game_over(self) -> None:
        if not self.board.is_game_over():
            return
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            game_over_text = f"Checkmate! {winner} wins!"
        elif self.board.is_stalemate():
            game_over_text = "Stalemate!"
        elif self.board.is_insufficient_material():
            game_over_text = "Draw: Insufficient material!"
        elif self.board.is_seventyfive_moves():
            game_over_text = "Draw: 75-moves rule!"
        elif self.board.is_fivefold_repetition():
            game_over_text = "Draw: Fivefold repetition!"
        elif self.board.is_variant_end():
            game_over_text = "Game variant end!"
        elif self.board.is_game_over(claim_draw=True):
            game_over_text = "Draw!"
        else:
            game_over_text = "Game Over!"

        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))
        text_surface = game_over_font.render(game_over_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        background_rect = text_rect.inflate(20, 20)
        pygame.draw.rect(screen, (50, 50, 50), background_rect)
        pygame.draw.rect(screen, (200, 200, 200), background_rect, 2)
        screen.blit(text_surface, text_rect)

    def ponder_function(self, ponder_board: chess.Board, ponder_move_start: chess.Move) -> None:
        self.pondering_move = self.get_best_move(MAX_AI_DEPTH - 1)
        print(f"Pondering on opponent move: {ponder_move_start}, AI response: {self.pondering_move}")

    def ai_move(self) -> None:
        if not self.player_turn and not self.board.is_game_over():
            start_time = time.time()
            deadline = start_time + self.time_limit

            if self.pondering_move and self.board.move_stack and self.board.peek() == self.pondering_move:
                best_move = self.pondering_move
                print("Ponder hit!")
            else:
                best_move = self.get_best_move(MAX_AI_DEPTH)

            if best_move:
                self.board.push(best_move)
                self.start_pondering()
            else:
                self.stop_pondering()
            self.player_turn = True

    def start_pondering(self) -> None:
        if not self.board.is_game_over() and self.board.turn == chess.BLACK:
            possible_player_moves = list(self.board.legal_moves)
            if possible_player_moves:
                ponder_move_start = possible_player_moves[0]
                ponder_board = self.board.copy()
                ponder_board.push(ponder_move_start)
                if not ponder_board.is_game_over():
                    self.stop_pondering()
                    self.pondering_thread = threading.Thread(target=self.ponder_function, args=(ponder_board, ponder_move_start))
                    self.pondering_thread.start()

    def stop_pondering(self) -> None:
        if self.pondering_thread and self.pondering_thread.is_alive():
            pass
        self.pondering_move = None
        self.pondering_thread = None

    def update(self) -> None:
        if not self.player_turn:
            self.ai_move()

    def render(self) -> None:
        self.draw_board()
        self.highlight_moves()
        self.highlight_square(self.selected_square)
        self.draw_pieces()
        self.draw_turn_info()
        self.draw_game_over()
        pygame.display.flip()

    def run(self) -> None:
        running = True
        if self.board.turn == chess.BLACK:
            self.start_pondering()
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and self.player_turn:
                    self.handle_mouse_click(pygame.mouse.get_pos())
            self.update()
            self.render()
            clock.tick(60)
        pygame.quit()

def main() -> None:
    game = ChessGame()
    game.run()

if __name__ == "__main__":
    main()