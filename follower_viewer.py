"""Local read-only telemetry server. Never opens a robot serial port."""
import json
import math
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class FollowerViewer:
    def __init__(self, port=8765, open_browser=True):
        self.lock = threading.Lock()
        self.state = {"positions": {}, "timestamp": None, "status": "Waiting for robot", "source": "actual"}
        owner = self
        root = Path(__file__).parent / "viewer"

        class Handler(SimpleHTTPRequestHandler):
            extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                              ".js": "text/javascript", ".urdf": "application/xml"}
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root), **kwargs)

            def do_GET(self):
                if self.path == "/state":
                    with owner.lock:
                        data = json.dumps(owner.state, allow_nan=False).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    super().do_GET()

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        print(f"Follower viewer: {self.url}")
        if open_browser:
            webbrowser.open(self.url)

    def publish(self, positions, status="Moving", source="actual", command_positions=None, diagnostics=None, diagnostics_timestamp=None):
        positions = {k: float(v) for k, v in positions.items()}
        if not all(math.isfinite(value) for value in positions.values()):
            raise ValueError('Nonfinite follower position')
        with self.lock:
            self.state = {"positions": positions,
                          "timestamp": time.time(), "status": status, "source": source,
                          "feedback": {"command_positions": command_positions,
                                       "tracking_error": ({k: positions[k]-float(v) for k,v in command_positions.items() if k in positions}
                                                          if command_positions is not None else None),
                                       "diagnostics": diagnostics,
                                       "diagnostics_timestamp": (diagnostics_timestamp if diagnostics_timestamp is not None else time.time())
                                       if diagnostics is not None else None}}

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
