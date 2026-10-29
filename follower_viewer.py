"""Local read-only telemetry server. Never opens a robot serial port."""
import json
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

    def publish(self, positions, status="Moving", source="actual"):
        with self.lock:
            self.state = {"positions": {k: float(v) for k, v in positions.items()},
                          "timestamp": time.time(), "status": status, "source": source}

    def sample(self, bus, status="Moving"):
        # Called only by the control thread; no competing serial reads.
        self.publish(bus.sync_read("Present_Position", num_retry=2), status)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
