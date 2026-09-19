"""Play chess against the FlyWire connectome.

The fly does not understand chess. It has no board, no rules, no lookahead --
139,248 neurons wired for smelling, walking and courtship. What it can do is
respond differently to different stimuli, and that is all this uses.

Each legal move is encoded as a stimulus: the from-square drives one block of
sensory neurons, the to-square another. The whole brain runs for 150 ms and
the total spike count in the descending population is the move's score. The
highest score wins.

So every move really is a readout of the connectome, and the connectome
really does pick it -- but it is picking on neural excitability, not on
whether the move is any good. Expect to win.

    pip install chess
    python -m flybrain.chess_fly
    python -m flybrain.chess_fly --ms 250     # longer look, slower
"""
import sys

import numpy as np

try:
    import chess
except ImportError:
    sys.exit("needs python-chess:  pip install chess")

from .brain import Brain, Params

N_SQUARES = 64
DRIVE_HZ = 150.0


class FlyPlayer:
    def __init__(self, ann, W, ms=150.0):
        self.ann = ann
        self.W = W
        self.seconds = ms / 1000.0
        probe = Brain(ann, W, Params())
        # Partition the sensory population into 64 blocks, one per square.
        # Which neurons land in which block is arbitrary -- it has to be,
        # there is nothing chess-shaped in a fly -- but it is fixed, so the
        # same position always produces the same stimulus.
        # Use sensory AND optic cells: 264 neurons per square was too few to
        # push anything downstream and every move scored an identical zero.
        pool = np.union1d(probe.select(super_class="sensory"),
                          probe.select(super_class="optic"))
        rng = np.random.default_rng(12345)
        pool = rng.permutation(pool)      # fixed shuffle, so blocks mix regions
        self.blocks = np.array_split(pool, N_SQUARES)
        self.readout = probe.select(super_class="descending")
        print(f"stimulus pool: {len(pool):,} neurons over 64 squares "
              f"(~{len(pool)//64} each)")
        print(f"readout: {self.readout.size} descending neurons\n")

    def score(self, move):
        """Return (descending Hz, whole-brain Hz) for this move's stimulus."""
        brain = Brain(self.ann, self.W, Params())
        stim = {}
        for sq in (move.from_square, move.to_square):
            for i in self.blocks[sq]:
                stim[int(i)] = DRIVE_HZ
        rates = brain.run(self.seconds, stim, np.random.default_rng(0))
        return float(rates[self.readout].sum()), float(rates.sum())

    def pick(self, board, verbose=False):
        """Return (chosen move, [{san, descending, whole_brain}, ...])."""
        moves = list(board.legal_moves)
        scored = []
        for n, mv in enumerate(moves, 1):
            dn, tot = self.score(mv)
            scored.append((dn, tot, mv))
            print(f"\r  thinking {n}/{len(moves)} ...", end="", flush=True)
        print("\r" + " " * 32 + "\r", end="")

        # Rank on descending activity, breaking ties on whole-brain activity.
        scored.sort(key=lambda t: (-t[0], -t[1], t[2].uci()))

        n_dn = len({round(t[0], 3) for t in scored})
        n_tot = len({round(t[1], 3) for t in scored})
        if n_dn == 1 and n_tot == 1:
            print("  the connectome gives every legal move an identical "
                  "response --\n  it cannot distinguish them, so this move is "
                  "arbitrary.")
        elif verbose:
            print(f"  {n_dn} distinct descending responses across "
                  f"{len(moves)} moves:")
            for dn, tot, mv in scored[:4]:
                print(f"    {board.san(mv):<8} descending {dn:8.0f} Hz"
                      f"   whole brain {tot:9.0f} Hz")
        detail = [{"san": board.san(mv), "uci": mv.uci(),
                   "descending": round(dn, 1), "whole": round(tot, 1)}
                  for dn, tot, mv in scored[:6]]
        return scored[0][2], {"candidates": detail, "distinct": n_dn,
                              "moves": len(moves)}


def main():
    argv = sys.argv[1:]
    ms = float(argv[argv.index("--ms") + 1]) if "--ms" in argv else 150.0

    from .data import load
    ann, W, _ = load()
    fly = FlyPlayer(ann, W, ms)
    board = chess.Board()

    print("You are White. Enter moves as e2e4 or Nf3. 'quit' to stop.\n")
    while not board.is_game_over():
        print(board.unicode(invert_color=True, empty_square="."))
        print()
        text = input("your move: ").strip()
        if text in {"quit", "exit"}:
            return 0
        try:
            move = board.parse_san(text)
        except ValueError:
            try:
                move = chess.Move.from_uci(text)
                if move not in board.legal_moves:
                    raise ValueError
            except ValueError:
                print("  not a legal move, try again\n")
                continue
        board.push(move)
        if board.is_game_over():
            break

        print("\nthe fly is deciding:")
        reply, _ = fly.pick(board, verbose=True)
        print(f"  fly plays {board.san(reply)}\n")
        board.push(reply)

    print(board.unicode(invert_color=True, empty_square="."))
    print(f"\ngame over: {board.result()}  ({board.outcome().termination.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
