#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import queue
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import websockets
import websockets.asyncio.server

from model import bot_script, ImprovisingScript
from value_model import ValueNet, ImprovisingValueNet

#logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

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


class HTTPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global _mode, _no_boost
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == '/':
            if 'mode' in body:
                if body['mode'] in ('script', 'iscript', 'value', 'ivalue', 'record'):
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


class ReuseHTTPServer(HTTPServer):
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


_iscript = ImprovisingScript()
_value = ValueNet()
_ivalue = ImprovisingValueNet(lambda x: _value.act(x))

_mode = 'iscript'
modes = dict(
    iscript=lambda x: _iscript.act(x),
    ivalue=lambda x: _ivalue.act(x),
    value=lambda x: (_value.act(x), None),
    script=lambda x: (bot_script(x), None),
    record=lambda f: ([-1, 0], None),
)

async def ws_handler(websocket):
    global _no_boost
    print(f"[server] new connection — mode={_mode}  no_boost={_no_boost}")
    episode_buf = []

    async for message in websocket:
        try:
            state = json.loads(message)
            if not state:
                for frame in episode_buf:
                    _log(frame)
                _log(json.dumps(state))
                episode_buf.clear()
                _roll_no_boost()
                print(f"[server] DEAD — no_boost={_no_boost} active={_no_boost_active}\n")
            else:
                action, improv = modes[_mode](state)
                if _no_boost_active and action:
                    action[1] = 0
                episode_buf.append(json.dumps([state, action, improv]))
                await websocket.send(json.dumps([action[0], action[1], state[-1]])) # pass timestamp through

        except Exception as e:
            import traceback, sys
            traceback.print_exc()
            _debug_done.clear()
            _debug_queue.put(sys.exc_info())
            await asyncio.get_event_loop().run_in_executor(None, _debug_done.wait)
            await websocket.send(json.dumps({"error": str(e)}))


class _PNAServerConnection(websockets.asyncio.server.ServerConnection):
    """Intercept Chrome's Private Network Access OPTIONS preflight before websockets parses it."""
    def data_received(self, data: bytes) -> None:
        if data.lstrip().startswith(b"OPTIONS"):
            self.transport.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Access-Control-Allow-Origin: *\r\n"
                b"Access-Control-Allow-Private-Network: true\r\n"
                b"Content-Length: 0\r\n"
                b"\r\n"
            )
            self.transport.close()
        else:
            super().data_received(data)


async def _process_response(connection, request, response):
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


async def run_ws():
    async with websockets.serve(ws_handler, "", WS_PORT,
                                create_connection=_PNAServerConnection,
                                process_response=_process_response):
        print(f"WebSocket on ws://localhost:{WS_PORT}")
        print(f"HTTP control on http://localhost:{HTTP_PORT}/mode")
        await asyncio.get_event_loop().create_future()


def main():
    threading.Thread(target=_debug_worker, daemon=True).start()

    threading.Thread(
        target=lambda: ReuseHTTPServer(('', HTTP_PORT), HTTPHandler).serve_forever(),
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
        watch_paths = [p for p in [__file__, "server.py", "model.pt", "value.pt", "value2.pt"] if os.path.exists(p)]
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
