"""3D chess against the connectome, in a browser.

The page is only a renderer and an input device. Every fly move is computed
here, by running the whole 139,248-neuron brain once per legal move -- the
browser could not hold the connectome, and a fake one would defeat the point.

    pip install chess
    python -m flybrain.server
    then open http://localhost:8000

    --ms 250        longer simulation per move (slower, more discriminating)
    --port 8123     different port
    --synthetic     random wiring, for testing the UI without the real data
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import chess
except ImportError:
    sys.exit("needs python-chess:  pip install chess")

WEB = Path(__file__).parent / "web"

state = {"board": chess.Board(), "fly": None, "bot": None,
         "opponent": "fly", "mode": "play", "last": None}
lock = threading.Lock()


def snapshot(extra=None):
    b = state["board"]
    out = {
        "fen": b.fen(),
        "turn": "white" if b.turn == chess.WHITE else "black",
        "legal": [m.uci() for m in b.legal_moves],
        "check": b.is_check(),
        "over": b.is_game_over(),
        "result": b.result() if b.is_game_over() else None,
        "last": state["last"],
        "history": [m.uci() for m in b.move_stack],
        "opponent": state["opponent"],
        "mode": state["mode"],
    }
    if extra:
        out.update(extra)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, (WEB / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        elif path == "/state":
            with lock:
                self._send(200, snapshot())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        path = self.path.split("?")[0]

        if path == "/new":
            with lock:
                state["board"] = chess.Board()
                state["last"] = None
                if "opponent" in req:
                    state["opponent"] = req["opponent"]
                if "mode" in req:
                    state["mode"] = req["mode"]
                self._send(200, snapshot())
            return

        if path == "/config":
            with lock:
                if "opponent" in req:
                    state["opponent"] = req["opponent"]
                if "mode" in req:
                    state["mode"] = req["mode"]
                self._send(200, snapshot())
            return

        if path == "/step":
            # watch mode: advance whichever side is to move, one ply
            with lock:
                board = state["board"]
                if board.is_game_over():
                    self._send(200, snapshot())
                    return
                who = "bot" if board.turn == chess.WHITE else "fly"
                mover = state[who]
                mv, detail = mover.pick(board)
                san = board.san(mv)
                board.push(mv)
                state["last"] = mv.uci()
                self._send(200, snapshot({
                    "mover": who, "moverSan": san, "moverUci": mv.uci(),
                    "detail": detail}))
            return

        if path == "/move":
            with lock:
                board = state["board"]
                try:
                    move = chess.Move.from_uci(req.get("uci", ""))
                except ValueError:
                    self._send(400, {"error": "bad move"})
                    return
                if move not in board.legal_moves:
                    self._send(400, {"error": "illegal move"})
                    return
                san = board.san(move)
                board.push(move)
                state["last"] = move.uci()
                if board.is_game_over():
                    self._send(200, snapshot({"yourSan": san, "flySan": None}))
                    return
                who = state["opponent"]
                reply, detail = state[who].pick(board)
                reply_san = board.san(reply)
                board.push(reply)
                state["last"] = reply.uci()
                self._send(200, snapshot({
                    "yourSan": san, "flySan": reply_san,
                    "flyUci": reply.uci(), "mover": who, "detail": detail,
                }))
            return

        self._send(404, {"error": "not found"})


def main():
    argv = sys.argv[1:]
    ms = float(argv[argv.index("--ms") + 1]) if "--ms" in argv else 120.0
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8000

    from .chess_fly import FlyPlayer
    if "--synthetic" in argv:
        from .synthetic import make
        ann, W, _ = make()
    else:
        from .data import load
        ann, W, _ = load()
    state["fly"] = FlyPlayer(ann, W, ms)
    from .bot import Bot
    state["bot"] = Bot(depth=3)

    print(f"\n  open  http://localhost:{port}\n")
    print("  Each fly move runs the whole brain once per legal move,")
    print("  so expect roughly 10-20 seconds of thinking per turn.\n")
    # Threading matters: Chrome opens several connections at once and a
    # single-threaded HTTPServer serialises them, so the page hangs blank.
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
