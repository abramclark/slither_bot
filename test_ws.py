#!/usr/bin/env python3
import asyncio
import json
import websockets

async def test():
    with open('test_msg.json') as f:
        raw = f.read()

    print("raw start:", repr(raw[:30]))
    print("raw end:  ", repr(raw[-30:]))

    # strip surrounding single quotes if present
    stripped = raw.strip()
    if stripped.startswith("'") and stripped.endswith("'"):
        stripped = stripped[1:-1]
        print("stripped surrounding single quotes")

    try:
        parsed = json.loads(stripped)
        print("json.loads ok, type:", type(parsed), "len:", len(parsed))
    except Exception as e:
        print("json.loads error:", e)
        return

    async with websockets.connect("ws://localhost:9002") as ws:
        await ws.send(stripped)
        response = await ws.recv()
        print("response:", response)

asyncio.run(test())
