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
from websockets.http11 import Response
from websockets.datastructures import Headers

from environment import bot_script, ImprovisingScript
from policy_model import PolicyNet
from sac_model import SACNet

#logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

WS_PORT   = 9002
HTTP_PORT = 9001

_debug_queue = queue.Queue()
_debug_done  = threading.Event()
_no_boost        = False  # True / False / 'random'
_no_boost_active = random.random() < 0.5
_session_count   = 0


def _roll_no_boost():
    global _no_boost_active
    if _no_boost == 'random':
        _no_boost_active = random.random() < 0.5
    else:
        _no_boost_active = bool(_no_boost)
    print(f"[server] current no_boost → {_no_boost_active}")


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
                if body['mode'] in ('script', 'iscript', 'policy', 'sac', 'record'):
                    _mode = body['mode']
                    print(f"[server] mode → {_mode}")
                else:
                    self.send_response(400)
                    self.end_headers()
                    return
            if 'no_boost' in body:
                val = body['no_boost']
                _no_boost = 'random' if val == 'random' else bool(val)
                print(f"[server] no_boost → {_no_boost}")
            if 'det' in body:
                _sac.deterministic = bool(body['det'])
                print(f"[server] sac.deterministic → {_sac.deterministic}")
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(_control_state()).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/':
            body = json.dumps(_control_state()).encode()
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




def _reopen_if_needed(f, path):
    try:
        if not os.path.exists(path) or os.fstat(f.fileno()).st_ino != os.stat(path).st_ino:
            f.close()
            return open(path, "a")
    except OSError:
        return open(path, "a")
    return f


def _control_state():
    return {
        'mode': _mode,
        'no_boost': _no_boost,
        'det': int(_sac.deterministic),
    }


_iscript = ImprovisingScript()
_policy = PolicyNet()
_sac = SACNet()

_mode = 'script'
modes = dict(
    script=lambda x: (bot_script(x), None),
    iscript=lambda x: _iscript.act(x),
    policy=lambda x: (_policy.act(x), None),
    sac=lambda x: (_sac.act(x), None),
    record=lambda f: ([-1, 0], None),
)

async def ws_handler(websocket):
    global _session_count
    session = websocket.request.path.lstrip('/')
    if not session:
        _session_count += 1
        session = str(_session_count)
    os.makedirs("experience", exist_ok=True)
    log_path = f"experience/{session}.jsonl"
    log_file = open(log_path, "a")
    episode_buf = []
    print(f"[server] new connection — session={session} mode={_mode}  no_boost={_no_boost}")

    try:
        async for message in websocket:
            try:
                state = json.loads(message)
                if not state:
                    log_file = _reopen_if_needed(log_file, log_path)
                    for frame in episode_buf:
                        log_file.write(frame + "\n")
                    log_file.write(json.dumps(state) + "\n")
                    log_file.flush()
                    episode_buf.clear()
                    print(f"[server] DEAD — session={session} no_boost={_no_boost}\n")
                    _roll_no_boost()
                else:
                    action, improv = modes[_mode](state)
                    if _no_boost_active and action:
                        action[1] = 0
                    episode_buf.append(json.dumps([state, action, improv]))
                    send = improv if improv else action
                    await websocket.send(json.dumps([send[0], send[1], state[-1]])) # pass timestamp through

            except Exception as e:
                import traceback, sys
                traceback.print_exc()
                _debug_done.clear()
                _debug_queue.put(sys.exc_info())
                await asyncio.get_event_loop().run_in_executor(None, _debug_done.wait)
                await websocket.send(json.dumps({"error": str(e)}))
    finally:
        log_file.close()


async def _process_request(connection, request):
    if request.headers.get('Upgrade', '').lower() != 'websocket':
        return Response(200, 'OK', Headers([
            ('Access-Control-Allow-Origin', '*'),
            ('Access-Control-Allow-Private-Network', 'true'),
            ('Content-Length', '0'),
        ]))

async def _process_response(connection, request, response):
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


async def run_ws():
    async with websockets.serve(ws_handler, "", WS_PORT,
                                process_request=_process_request,
                                process_response=_process_response):
        print(f"WebSocket on ws://localhost:{WS_PORT}")
        print(f"HTTP control on http://localhost:{HTTP_PORT}/mode")
        await asyncio.get_event_loop().create_future()


def main():
    _policy.load()
    _sac.load()
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
        watch_names = {"server.py", "policy.pt", "sac.pt"}
        try:
            for changes in watch("."):
                if not any(os.path.basename(path) in watch_names for _, path in changes):
                    continue
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
