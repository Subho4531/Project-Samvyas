"""Run the local Samvyas browser recorder."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from samvyas.web.voice_recorder import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PROMPTS_PATH,
    VoiceRecorderService,
    make_server,
)


def main() -> int:
    """Serve the local-only recorder until Ctrl+C."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Local bind address.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-root", type=Path, default=ROOT / DEFAULT_DATA_ROOT)
    parser.add_argument("--prompts", type=Path, default=ROOT / DEFAULT_PROMPTS_PATH)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("For privacy, the recorder only binds to a loopback address.")
    try:
        service = VoiceRecorderService(args.data_root, args.prompts)
        server = make_server(args.host, args.port, service)
    except (OSError, ValueError) as exc:
        print(f"Could not start recorder: {exc}", file=sys.stderr)
        return 1
    address = "localhost" if args.host == "localhost" else args.host
    url = f"http://{address}:{server.server_port}"
    print(f"Samvyas Voice Studio: {url}")
    print(f"Samples are stored only under: {service.data_root}")
    print("Press Ctrl+C to stop the local server.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nVoice Studio stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
