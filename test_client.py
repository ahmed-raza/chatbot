import asyncio
import sys
import websockets

async def test_client():
    uri = "ws://localhost:8765"

    try:
        async with websockets.connect(uri) as websocket:
            # Send first message
            await websocket.send("Why is the earth round")

            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5)  # Wait for response
                    sys.stdout.write(f"{message}")
                    sys.stdout.flush()
                except asyncio.TimeoutError:
                    break
            print(f"\n")
            # Send second message
            await websocket.send("What is my name?")

            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5)  # Wait for response
                    sys.stdout.write(f"{message}")
                    sys.stdout.flush()
                except asyncio.TimeoutError:
                    break

    except websockets.ConnectionClosed as e:
        print(f"❌ Connection closed: {e}")
    except Exception as e:
        print(f"⚠️ Error: {e}")

asyncio.run(test_client())
