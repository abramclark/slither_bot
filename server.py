#!/usr/bin/env python3
import asyncio
import json
import os
import queue
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
import websockets

from model import get_flat, bot_script
from time import time
from value_model import ValueNet
from value2_model import Value2Net

WS_PORT   = 9002
HTTP_PORT = 9001
LOG_PATH  = "experience.jsonl"

_debug_queue = queue.Queue()
_debug_done  = threading.Event()
_no_boost        = False  # True / False / 'random'
_no_boost_active = random.random() < 0.5


def _roll_no_boost():
    global _no_boost_active
    _no_boost_active = random.random() < 0.5 if _no_boost == 'random' else bool(_no_boost)


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
        global _mode, _no_boost
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == '/':
            if 'mode' in body:
                if body['mode'] in ('value', 'value2', 'script', 'record'):
                    _mode = body['mode']
                    print(f"[server] mode → {_mode}")
                else:
                    self.send_response(400)
                    self.end_headers()
                    return
            if 'no_boost' in body:
                val = body['no_boost']
                _no_boost = 'random' if val == 'random' else bool(val)
                _roll_no_boost()
                print(f"[server] no_boost → {_no_boost}")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'mode': _mode, 'no_boost': _no_boost}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/':
            body = json.dumps({'mode': _mode, 'no_boost': _no_boost, 'no_boost_current': _no_boost_active}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # suppress per-request logs


class _ReuseHTTPServer(HTTPServer):
    allow_reuse_address = True


_log_file = open(LOG_PATH, "a")
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


class ValueRuntime:
    def __init__(self, model_cls):
        save_path = model_cls.save_path
        self.tag = model_cls.__name__
        self.model = model_cls()
        try:
            ckpt = torch.load(save_path, weights_only=True)
            self.model.load_state_dict(ckpt["model"])
            ep = ckpt.get("ep", 0)
            print(f"{self.tag}: resumed from {save_path} (ep={ep})")
        except FileNotFoundError:
            print(f"{self.tag}: no checkpoint at {save_path}, starting fresh")
        self.model.eval()

    def handle_message(self, state_d):
        if not state_d:
            return [0, 0]
        x = get_flat(state_d).astype(np.float32)
        game_dir, boost, val = self.model.act(x)
        if state_d[0] < 3: boost = 0
        print(f"[{self.tag}] dir={game_dir:.3f}  boost={boost}  val={val:.3f}")
        return [game_dir, boost]


_value_runtime = _value2_runtime = None
def value_act(state):  return _value_runtime.handle_message(state)
def value2_act(state): return _value2_runtime.handle_message(state)


_mode = 'value'
modes = dict(
    value=value_act,
    value2=value2_act,
    script=lambda s: bot_script(s)[:2],
    record=lambda f: [-1, 0],
)


async def ws_handler(websocket):
    global _no_boost
    print(f"[server] new connection — mode={_mode}  no_boost={_no_boost}")

    episode_buf = []

    async for message in websocket:
        try:
            state_d = json.loads(message)
            if not state_d:
                for frame in episode_buf:
                    _log(frame)
                _log(json.dumps(state_d))
                episode_buf.clear()
                _roll_no_boost()
                print(f"[server] DEAD — no_boost={_no_boost} active={_no_boost_active}\n")
            else:
                episode_buf.append(json.dumps(state_d))
            response = modes[_mode](state_d)
            if _no_boost_active and response:
                response[1] = 0
            await websocket.send(json.dumps(response))

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
    global _value_runtime, _value2_runtime
    _value_runtime  = ValueRuntime(ValueNet)
    _value2_runtime = ValueRuntime(Value2Net)

    threading.Thread(target=_debug_worker, daemon=True).start()

    threading.Thread(
        target=lambda: _ReuseHTTPServer(('', HTTP_PORT), _ControlHandler).serve_forever(),
        daemon=True,
    ).start()

    asyncio.run(run_ws())


if __name__ == "__main__":
    import sys, subprocess
    print(f"[server] no_boost={_no_boost}")
    if "--no-reload" in sys.argv:
        main()
    else:
        from watchfiles import watch
        cmd = [sys.executable, __file__, "--no-reload"]
        proc = subprocess.Popen(cmd)
        watch_paths = [p for p in [__file__, "model.py", "value_model.py", "value2_model.py", "value.pt", "value2.pt"] if os.path.exists(p)]
        try:
            for _ in watch(*watch_paths):
                print("File changed — restarting...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                proc = subprocess.Popen(cmd)
        except KeyboardInterrupt:
            pass
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
