"""Record the terminal demo: run a few offline commands (mock models, no key, no network), capture
their real output with timings, and write

* ``docs/demo/demo.txt``  -- the transcript,
* ``docs/demo/demo.cast`` -- an asciicast v2 recording (play with ``asciinema play``),
* ``docs/demo/demo.gif``  -- an animated GIF rendered from the same capture (needs Pillow:
  ``pip install pillow`` inside the virtual environment; skipped with a message otherwise).

    python scripts/demo.py

Every frame shows output the commands actually printed; nothing is typed by hand or edited.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"
COMMANDS = [
    "python examples/06_reflection.py --mock",
    "python examples/04_planned.py --mock",
    "python -m trellis_agent.eval run evals/smoke --llm mock --prices evals/prices.demo.yaml "
    "--out runs/demo.json",
    "python -m trellis_agent.eval compare evals/baselines/smoke.json runs/demo.json",
]
COLS, ROWS = 96, 26
PROMPT = "$ "


def capture() -> list[tuple[float, str]]:
    """(seconds since start, text) chunks: the prompt and command, then each output line."""
    events: list[tuple[float, str]] = []
    started = time.perf_counter()
    for command in COMMANDS:
        events.append((time.perf_counter() - started, f"{PROMPT}{command}\n"))
        argv = shlex.split(command)
        argv[0] = sys.executable
        done = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
        for line in (done.stdout + done.stderr).splitlines():
            events.append((time.perf_counter() - started, line + "\n"))
        events.append((time.perf_counter() - started, "\n"))
    return events


def write_cast(events: list[tuple[float, str]], path: Path) -> None:
    header = {"version": 2, "width": COLS, "height": ROWS, "title": "Trellis offline demo"}
    lines = [json.dumps(header)] + [json.dumps([round(t, 3), "o", s.replace("\n", "\r\n")])
                                    for t, s in events]  # fmt: skip
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def wrap(text: str) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        out += [line[i : i + COLS] for i in range(0, max(len(line), 1), COLS)]
    return out


def write_gif(events: list[tuple[float, str]], path: Path) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow is not installed: skipped the GIF (pip install pillow in the venv)")
        return False
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except OSError:
        font = ImageFont.load_default()
    cell_w = int(font.getlength("M")) or 8
    cell_h = 18
    size = (COLS * cell_w + 20, ROWS * cell_h + 20)
    screen: list[str] = []
    frames, durations = [], []

    def frame(hold_ms: int) -> None:
        image = Image.new("P", size, 0)
        image.putpalette([24, 24, 27, 228, 228, 231, 134, 239, 172] + [0] * (256 - 3) * 3)
        draw = ImageDraw.Draw(image)
        for row, line in enumerate(screen[-ROWS:]):
            color = 2 if line.startswith(PROMPT) else 1
            draw.text((10, 10 + row * cell_h), line, font=font, fill=color)
        frames.append(image)
        durations.append(hold_ms)

    pending = 0
    for _, text in events:
        is_command = text.startswith(PROMPT)
        if is_command and pending:
            frame(1500)  # let the previous command's output be read
            pending = 0
        screen.extend(wrap(text.rstrip("\n")))
        pending += 1
        if is_command or pending >= 3:
            frame(900 if is_command else 150)
            pending = 0
    frame(4000)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True)  # fmt: skip
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = capture()
    (OUT / "demo.txt").write_text("".join(t for _, t in events), encoding="utf-8")
    write_cast(events, OUT / "demo.cast")
    made = write_gif(events, OUT / "demo.gif")
    print(
        f"wrote {OUT / 'demo.txt'}, {OUT / 'demo.cast'}" + (f", {OUT / 'demo.gif'}" if made else "")
    )


if __name__ == "__main__":
    main()
