#!/usr/bin/env python3
import asyncio
import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import websockets

from runtime import Session, SurvivalSession, ValueSession, get_runtime, get_survival_runtime, get_value_runtime

WS_PORT   = 9002
HTTP_PORT = 9001
LOG_PATH  = "experience.jsonl"

_debug_queue = queue.Queue()
_debug_done  = threading.Event()
_mode = 'value'  # 'model', 'survival', or 'value'


def _debug_worker():
    while True:
        exc_type, exc_value, tb = _debug_queue.get()
        import rpdb
        print("Debugger listening on port 4444 — connect with: telnet localhost 4444")
        debugger = rpdb.get_debugger_class()()
        debugger.reset()
        debugger.interaction(None, tb)
        _debug_done.set()


class _ControlHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global _mode
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == '/mode':
            new_mode = body.get('mode', '')
            if new_mode in ('model', 'survival', 'value'):
                _mode = new_mode
                print(f"[server] mode → {_mode}")
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'mode': _mode}).encode())
            else:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # suppress per-request logs


threading.Thread(target=_debug_worker, daemon=True).start()
class _ReuseHTTPServer(HTTPServer):
    allow_reuse_address = True

threading.Thread(
    target=lambda: _ReuseHTTPServer(('', HTTP_PORT), _ControlHandler).serve_forever(),
    daemon=True,
).start()

runtime          = get_runtime()
survival_runtime = get_survival_runtime()
value_runtime    = get_value_runtime()


def _log(line):
    """Append a line to the log, reopening the file if it has been moved or deleted."""
    global _log_file
    try:
        if not os.path.exists(LOG_PATH) or os.fstat(_log_file.fileno()).st_ino != os.stat(LOG_PATH).st_ino:
            _log_file.close()
            _log_file = open(LOG_PATH, "a")
    except OSError:
        _log_file = open(LOG_PATH, "a")
    _log_file.write(line + "\n")
    _log_file.flush()


_log_file = open(LOG_PATH, "a")


async def ws_handler(websocket):
    if _mode == 'survival':
        session = SurvivalSession(survival_runtime)
    elif _mode == 'value':
        session = ValueSession(value_runtime)
    else:
        session = Session(runtime)
    print(f"[server] new connection — mode={_mode}")

    async for message in websocket:
        try:
            state_d = json.loads(message)
            _log(json.dumps(state_d))
            await websocket.send(json.dumps(session.handle_message(state_d)))

        except Exception as e:
            import traceback, sys
            traceback.print_exc()
            _debug_done.clear()
            _debug_queue.put(sys.exc_info())
            await asyncio.get_event_loop().run_in_executor(None, _debug_done.wait)
            await websocket.send(json.dumps({"error": str(e)}))


async def run_ws():
    async with websockets.serve(ws_handler, "", WS_PORT):
        print(f"WebSocket on ws://localhost:{WS_PORT}")
        print(f"HTTP control on http://localhost:{HTTP_PORT}/mode")
        await asyncio.get_event_loop().create_future()


def main():
    asyncio.run(run_ws())


if __name__ == "__main__":
    import sys, subprocess
    if "--no-reload" in sys.argv:
        main()
    else:
        from watchfiles import watch
        cmd = [sys.executable, __file__, "--no-reload"]
        proc = subprocess.Popen(cmd)
        try:
            for _ in watch(__file__, "model.py", "runtime.py", "survival_model.py", "value_model.py",
                           "model.pt", "survival.pt", "value.pt"):
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
