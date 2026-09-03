"""Capture the Mac microphone and stream it to the ASR server.

Prints live captions: committed text bright, unstable tail dim.

Run:  python tools/mic_client.py [--url ws://localhost:8765/ws]
"""

import argparse
import asyncio
import json
import queue
import sys

import sounddevice as sd
import websockets

SAMPLE_RATE = 16000
BLOCK_SAMPLES = 800  # 50 ms of audio per websocket frame

DIM = "\033[2m"
GREEN = "\033[32m"
RESET = "\033[0m"


def render(committed: str, tail: str, translation: str, final: bool) -> None:
    guj = f"{committed} {DIM}{tail}{RESET}" if tail else committed
    sys.stdout.write("\r\033[2K" + guj + "\n\033[2K" + f"{GREEN}{translation}{RESET}")
    # stay on the English line and jump back up so the pair updates in place
    sys.stdout.write("\n\n" if final else "\033[1A\r")
    sys.stdout.flush()


async def run(url: str) -> None:
    audio_q: queue.Queue[bytes] = queue.Queue()

    def on_audio(indata, frames, time_info, status) -> None:
        if status:
            print(f"\naudio status: {status}", file=sys.stderr)
        audio_q.put(bytes(indata))

    async with websockets.connect(url, max_size=None) as ws:
        print(f"connected to {url} — speak (ctrl-c to stop)\n")

        async def sender() -> None:
            loop = asyncio.get_running_loop()
            while True:
                chunk = await loop.run_in_executor(None, audio_q.get)
                await ws.send(chunk)

        async def receiver() -> None:
            async for message in ws:
                event = json.loads(message)
                render(
                    event["committed"],
                    event["tail"],
                    event.get("translation", ""),
                    event["type"] == "final",
                )

        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SAMPLES,
            channels=1,
            dtype="int16",
            callback=on_audio,
        )
        with stream:
            await asyncio.gather(sender(), receiver())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8765/ws")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.url))
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
