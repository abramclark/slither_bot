#!/usr/bin/env python3
import asyncio
import json
import queue
import threading

import websockets

from runtime import Session, get_runtime

WS_PORT  = 9002
LOG_PATH = "experience.jsonl"

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


async def run_ws():
    async with websockets.serve(ws_handler, "", WS_PORT):
        print(f"WebSocket on ws://localhost:{WS_PORT}")
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
