"""A small conventional chess engine, as a control for the fly.

Negamax with alpha-beta, material plus piece-square tables, MVV-LVA move
ordering and a captures-only quiescence search. Perhaps 1200-1500 Elo --
nothing special, but it plays real chess, which is exactly the point: it is
what the fly is *not*.
"""
import chess

VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
         chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

PST = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
         0,  0,  0,  0,  0,  0,  0,  0],
    chess.KNIGHT: [
       -50,-40,-30,-30,-30,-30,-40,-50,
       -40,-20,  0,  5,  5,  0,-20,-40,
       -30,  5, 10, 15, 15, 10,  5,-30,
       -30,  0, 15, 20, 20, 15,  0,-30,
       -30,  5, 15, 20, 20, 15,  5,-30,
       -30,  0, 10, 15, 15, 10,  0,-30,
       -40,-20,  0,  0,  0,  0,-20,-40,
       -50,-40,-30,-30,-30,-30,-40,-50],
    chess.BISHOP: [
       -20,-10,-10,-10,-10,-10,-10,-20,
       -10,  5,  0,  0,  0,  0,  5,-10,
       -10, 10, 10, 10, 10, 10, 10,-10,
       -10,  0, 10, 10, 10, 10,  0,-10,
       -10,  5,  5, 10, 10,  5,  5,-10,
       -10,  0,  5, 10, 10,  5,  0,-10,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -20,-10,-10,-10,-10,-10,-10,-20],
    chess.ROOK: [
         0,  0,  5, 10, 10,  5,  0,  0,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         5, 10, 10, 10, 10, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0],
    chess.QUEEN: [
       -20,-10,-10, -5, -5,-10,-10,-20,
       -10,  0,  5,  0,  0,  0,  0,-10,
       -10,  5,  5,  5,  5,  5,  0,-10,
         0,  0,  5,  5,  5,  5,  0, -5,
        -5,  0,  5,  5,  5,  5,  0, -5,
       -10,  0,  5,  5,  5,  5,  0,-10,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -20,-10,-10, -5, -5,-10,-10,-20],
    chess.KING: [
        20, 30, 10,  0,  0, 10, 30, 20,
        20, 20,  0,  0,  0,  0, 20, 20,
       -10,-20,-20,-20,-20,-20,-20,-10,
       -20,-30,-30,-40,-40,-30,-30,-20,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30],
}

MATE = 100000


def evaluate(board):
    if board.is_checkmate():
        return -MATE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    score = 0
    for sq, piece in board.piece_map().items():
        v = VALUE[piece.piece_type]
        idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
        v += PST[piece.piece_type][idx]
        score += v if piece.color == board.turn else -v
    return score


def order(board):
    def key(m):
        if board.is_capture(m):
            victim = board.piece_at(m.to_square)
            attacker = board.piece_at(m.from_square)
            vv = VALUE[victim.piece_type] if victim else 100
            av = VALUE[attacker.piece_type] if attacker else 0
            return 10000 + vv - av // 10
        if m.promotion:
            return 9000
        return 0
    return sorted(board.legal_moves, key=key, reverse=True)


def quiesce(board, alpha, beta, depth=3):
    stand = evaluate(board)
    if depth == 0 or stand >= beta:
        return max(stand, alpha) if stand < beta else beta
    alpha = max(alpha, stand)
    for m in order(board):
        if not board.is_capture(m):
            continue
        board.push(m)
        score = -quiesce(board, -beta, -alpha, depth - 1)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def negamax(board, depth, alpha, beta):
    if board.is_game_over():
        return evaluate(board)
    if depth == 0:
        return quiesce(board, alpha, beta)
    best = -MATE * 2
    for m in order(board):
        board.push(m)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


class Bot:
    name = "bot"

    def __init__(self, depth=3):
        self.depth = depth

    def pick(self, board):
        best, best_move, scored = -MATE * 3, None, []
        for m in order(board):
            board.push(m)
            s = -negamax(board, self.depth - 1, -MATE * 3, MATE * 3)
            board.pop()
            scored.append((s, m))
            if s > best:
                best, best_move = s, m
        scored.sort(key=lambda t: -t[0])
        detail = {"candidates": [
            {"san": board.san(m), "uci": m.uci(),
             "descending": round(s / 100.0, 2), "whole": 0}
            for s, m in scored[:6]],
            "distinct": len({s for s, _ in scored}), "moves": len(scored),
            "unit": "pawns", "engine": "bot"}
        return best_move, detail
