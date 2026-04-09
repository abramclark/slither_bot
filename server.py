#!/usr/bin/env python3
"""
PPO-based RL server for slither bot.

WebSocket protocol (port 9002):
  browser → server:  JSON array [state, reward, done]
                     state = same format as record.jsonl rows
                     reward = float
                     done   = bool (true when snake died)
  server  → browser: {"dir": -1|0|1, "boost": 0|1}
"""
import asyncio
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import websockets

from runtime import Session, get_runtime

HTTP_PORT  = 9001
WS_PORT    = 9002
LOG_PATH   = "experience.jsonl"

# ---------------------------------------------------------------------------
# Debug thread — runs ipdb on main-like thread with proper terminal access
# ---------------------------------------------------------------------------

_debug_queue = queue.Queue()
_debug_done  = threading.Event()

def _debug_worker():
    while True:
        exc_type, exc_value, tb = _debug_queue.get()
        import rpdb
        print("Debugger listening on port 4444 — connect with: telnet localhost 4444")
        debugger = rpdb.get_debugger_class()()
        debugger.reset()
        debugger.interaction(None, tb)
        _debug_done.set()

threading.Thread(target=_debug_worker, daemon=True).start()
runtime = get_runtime()


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

_log_file = open(LOG_PATH, "a")

async def ws_handler(websocket):
    session = Session(runtime)

    async for message in websocket:
        try:
            state_d = json.loads(message)
            _log_file.write(json.dumps(state_d) + "\n")
            _log_file.flush()
            await websocket.send(json.dumps(session.handle_message(state_d)))

        except Exception as e:
            import traceback, sys
            traceback.print_exc()
            _debug_done.clear()
            _debug_queue.put(sys.exc_info())
            await asyncio.get_event_loop().run_in_executor(None, _debug_done.wait)
            await websocket.send(json.dumps({"error": str(e)}))

        #print(f'({time() - t0})')


async def run_ws():
    async with websockets.serve(ws_handler, "", WS_PORT):
        print(f"RL WebSocket on ws://localhost:{WS_PORT}")
        await asyncio.get_event_loop().create_future()


def start_ws():
    asyncio.run(run_ws())


# ---------------------------------------------------------------------------
# HTTP recording server (unchanged)
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code); self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_POST(self):
        print(f"[HTTP] POST {self.path!r}")
        length = int(self.headers.get("Content-Length", 0))
        mode   = json.loads(self.rfile.read(length))

        if self.path == "/training_mode":
            if mode not in ("ppo", "supervised"):
                self.send_response(400); self._cors(); self.end_headers()
                self.wfile.write(f"mode: {mode} must be 'ppo' or 'supervised'\n".encode()); return
            runtime.set_training_mode(mode)
            self._json(200, {"TRAINING_MODE": mode}); return

        if self.path == "/reset_optimizer":
            runtime.reset_optimizer()
            self._json(200, {"ok": True}); return

        self._json(404, {"error": f"unknown path: {self.path}"})

    def do_GET(self):
        self._json(200, runtime.get_config())

    def log_message(self, fmt, *args):
        pass


def main():
    threading.Thread(target=start_ws, daemon=True).start()
    server = HTTPServer(("", HTTP_PORT), Handler)
    print(f"HTTP on http://localhost:{HTTP_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    import sys, subprocess
    if "--no-reload" in sys.argv:
        main()
    else:
        from watchfiles import watch
        cmd = [sys.executable, __file__, "--no-reload"]
        proc = subprocess.Popen(cmd)
        try:
            for _ in watch(__file__, "model.py", "runtime.py"):
                print("File changed — restarting...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                proc = subprocess.Popen(cmd)
        except KeyboardInterrupt:
            proc.terminate()
