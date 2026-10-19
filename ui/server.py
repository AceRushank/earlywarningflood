"""
ui/server.py
============
Local HTTP server for the Vishnupuri Monsoon Timeline Dashboard.

Usage:
    python ui/server.py [--port 8080]

Serves the modern dashboard on http://localhost:8080
"""

import sys
import os
import json
import argparse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/config":
            carto_key = os.environ.get("CARTO_KEY", "")
            env_file = UI_DIR.parent / ".env"
            if not carto_key and env_file.exists():
                with open(env_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("CARTO_KEY="):
                            carto_key = line.split("=", 1)[1].strip()
                            break
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"carto_key": carto_key}).encode("utf-8"))
            return
        elif self.path.startswith("/api/live-status") or self.path.startswith("/api/live-telemetry"):
            try:
                repo_root = str(UI_DIR.parent)
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                from river_forecasting.src.live_inference import get_live_system_status
                bypass = "refresh=true" in self.path or "bypass=true" in self.path
                status_data = get_live_system_status(bypass_weather_cache=bypass)
                payload = json.dumps(status_data, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                err_payload = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_payload)))
                self.end_headers()
                self.wfile.write(err_payload)
            return
        elif self.path.startswith("/api/live-weather"):
            try:
                repo_root = str(UI_DIR.parent)
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                from river_forecasting.src.live_weather import fetch_open_meteo_live_data
                bypass = "refresh=true" in self.path or "bypass=true" in self.path
                weather_data = fetch_open_meteo_live_data(bypass_cache=bypass)
                payload = json.dumps(weather_data, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                err_payload = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_payload)))
                self.end_headers()
                self.wfile.write(err_payload)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        # Clean logging
        sys.stdout.write(f"[DASHBOARD] {self.address_string()} - {format % args}\n")
        sys.stdout.flush()


def run_server(port: int = 8080):
    server_address = ('127.0.0.1', port)
    try:
        httpd = ThreadingHTTPServer(server_address, DashboardHandler)
    except OSError:
        # Fallback to alternate port if busy
        port = 8081
        server_address = ('127.0.0.1', port)
        httpd = ThreadingHTTPServer(server_address, DashboardHandler)

    print("=" * 80)
    print("VISHNUPURI BARRAGE MONSOON TIMELINE DASHBOARD")
    print(f"Local Server running at: http://localhost:{port}")
    print(f"Serving files from: {UI_DIR}")
    print("Press Ctrl+C to terminate server.")
    print("=" * 80)
    sys.stdout.flush()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Vishnupuri Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    args = parser.parse_args()
    run_server(args.port)
