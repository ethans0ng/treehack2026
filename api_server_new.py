import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse, unquote

from results_writer import get_session, list_sessions, save_session


HOST = os.getenv("HGN_API_HOST", "0.0.0.0")
PORT = int(os.getenv("HGN_API_PORT", "8000"))
ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"
SCRIPT_DIR = ROOT / "script"


def _dashboard_html() -> str:
    template_path = TEMPLATES_DIR / "index.html"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>HGN Edge Dashboard</title>
  <link rel="stylesheet" href="/static/styles.css" />
</head>
<body>
  <main class="page-shell">
    <header class="hero">
      <p class="eyebrow">Stanford Police · Sobriety Operations</p>
     <h1>HGN Edge Dashboard</h1>
      <p class="hero-copy">Real-time operational view of edge-collected HGN sessions.</p>
    </header>
    <section class="card">
      <div class="row-head">
        <h2>Session history</h2>
        <div>
          <label for="limit">show</label>
          <select id="limit">
            <option value="10">10</option>
            <option value="25" selected>25</option>
            <option value="50">50</option>
          </select>
          <button id="refresh-btn">Refresh</button>
        </div>
      </div>
      <div id="latest-result">No test yet.</div>
      <div id="latest-warning"></div>
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Subject</th>
            <th>SP</th>
            <th>Prior 45°</th>
            <th>Max dev</th>
            <th>Vert</th>
            <th>Head warning</th>
          </tr>
        </thead>
        <tbody id="sessions-body"></tbody>
      </table>
    </section>
  </main>
  <script src="/static/appjs"></script>
</body>
</html>
"""


INLINE_APP_JS = """
fetchSessions();
async function fetchSessions() {
  const limit = Number(document.getElementById('limit').value || 25);
  const res = await fetch(`/api/sessions?limit=${encodeURIComponent(limit)}`);
  if (!res.ok) {
    document.getElementById('latest-result').textContent = `Error loading sessions (${res.status})`;
    return;
  }
  const data = await res.json();
  const sessions = data.items || [];

  const latestNode = document.getElementById('latest-result');
  const warningNode = document.getElementById('latest-warning');

  if (!sessions.length) {
    latestNode.textContent = 'No test yet.';
    warningNode.textContent = '';
  } else {
    const s = sessions[0];
    latestNode.textContent =
      `Latest: ${s.subject_name || 'Unknown'} at ${s.created_at} ` +
      `| SP L/R ${s.lack_of_smooth_pursuit_left_binary}/${s.lack_of_smooth_pursuit_right_binary} ` +
      `| Prior45 L/R ${s.nystagmus_prior_to_45_left_binary}/${s.nystagmus_prior_to_45_right_binary} ` +
      `| MaxDev L/R ${s.distinct_nystagmus_max_deviation_left_binary}/${s.distinct_nystagmus_max_deviation_right_binary}`;
    if (Number(s.head_warning_count || 0) >= 2) {
      warningNode.textContent = 'Head movement warning: HIGH. Result may be void.';
      warningNode.className = 'warning';
    } else {
      warningNode.textContent = 'Head movement warning: none.';
      warningNode.className = 'ok';
    }
  }

  const body = document.getElementById('sessions-body');
  body.innerHTML = '';
  sessions.forEach((s) => {
    const row = document.createElement('tr');
    const fields = [
      s.created_at || 'n/a',
      s.subject_name || 'Unknown',
      `${s.lack_of_smooth_pursuit_left_binary}/${s.lack_of_smooth_pursuit_right_binary}`,
      `${s.nystagmus_prior_to_45_left_binary}/${s.nystagmus_prior_to_45_right_binary}`,
      `${s.distinct_nystagmus_max_deviation_left_binary}/${s.distinct_nystagmus_max_deviation_right_binary}`,
      Number(s.vertical_nystagmus || 0).toFixed(1),
      Number(s.head_warning_count || 0) >= 2 ? 'void/retest' : 'ok',
    ];
    for (const field of fields) {
      const td = document.createElement('td');
      td.textContent = String(field);
      row.appendChild(td);
    }
    body.appendChild(row);
  });
}

document.getElementById('refresh-btn').addEventListener('click', fetchSessions);
fetchSessions();
"""

INLINE_STYLES_CSS = """
:root { --bg: #f4f6fb; --card: #ffffff; --ink: #111827; --muted: #4b5563; --line: #d1d5db; --accent: #2563eb; }
* { box-sizing: border-box; }
body { margin: 0; background: linear-gradient(160deg, #edf2ff 0%, #f7f7f7 45%, #fff 100%); color: var(--ink); font-family: "Trebuchet MS", "Segoe UI", sans-serif; padding: 24px; }
main { max-width: 1080px; margin: 0 auto; display: grid; gap: 16px; }
h1, h2 { margin: 0; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05); }
.row-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
table { width: 100%; border-collapse: collapse; }
th, td { border-bottom: 1px solid var(--line); text-align: left; padding: 8px; font-size: 14px; }
thead th { color: var(--muted); font-weight: 600; }
#latest-warning { margin-top: 10px; font-weight: 600; }
.warning { color: #dc2626; }
.ok { color: #16a34a; }
button { border: none; background: var(--accent); color: white; border-radius: 8px; padding: 6px 10px; cursor: pointer; }
select { border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px; }
"""


def _script_or_fallback(file_path: Path, fallback: str) -> str:
    if file_path.exists():
        try:
            return file_path.read_text(encoding="utf-8")
        except OSError:
            pass
    return fallback


def _embedded_static(path: str) -> str:
    if path in {"/static/app.js", "/app.js", "/appjs", "/static/appjs"}:
        return _script_or_fallback(SCRIPT_DIR / "app_new.js", INLINE_APP_JS)
    if path in {"/static/styles.css", "/styles.css"}:
        return _script_or_fallback(SCRIPT_DIR / "styles_new.css", INLINE_STYLES_CSS)

    relative = path.lstrip("/")
    static_path = ROOT / relative
    if static_path.exists():
        try:
            return static_path.read_text(encoding="utf-8")
        except OSError:
            pass
    return ""


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_static(handler: BaseHTTPRequestHandler, file_path: Path, content_type: str) -> None:
    with open(file_path, "rb") as f:
        data = f.read()
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_text(handler: BaseHTTPRequestHandler, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
    data = text.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _serve_path(path: str, handler: BaseHTTPRequestHandler) -> None:
    if ".." in path:
        handler.send_error(400, "Invalid path")
        return

    path = path.lstrip("/")
    if path == "":
        file_path = TEMPLATES_DIR / "index.html"
        content_type = "text/html; charset=utf-8"
    else:
        file_path = ROOT / path
        suffix = file_path.suffix.lower()
        if suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif suffix == ".json":
            content_type = "application/json; charset=utf-8"
        else:
            content_type = "application/octet-stream"

    if not file_path.exists():
        if path == "templates/index.html":
            _send_text(
                handler,
                "<!DOCTYPE html><html><head><meta charset='utf-8'><title>HGN Local API</title></head>"
                "<body><h1>HGN API is running</h1>"
                "<p>UI files are not found in this directory.</p>"
                "<ul>"
                "<li><a href=\"/api/sessions?limit=25\">/api/sessions</a></li>"
                "<li><a href=\"/api/session/demo\">/api/session/demo</a></li>"
                "</ul></body></html>",
                "text/html; charset=utf-8",
            )
        elif path.startswith("static/") and path.endswith(".js"):
            content = _embedded_static(f"/{path}")
            if content:
                _send_text(handler, content, "application/javascript; charset=utf-8")
            else:
                handler.send_error(404, "Not found")
        elif path.startswith("static/") and path.endswith(".css"):
            content = _embedded_static(f"/{path}")
            if content:
                _send_text(handler, content, "text/css; charset=utf-8")
            else:
                handler.send_error(404, "Not found")
        else:
            handler.send_error(404, "Not found")
        return
    _send_static(handler, file_path, content_type)


class HGNRequestHandler(BaseHTTPRequestHandler):
    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/")
        route = "/" if route == "" else route

        if route == "/":
            _send_text(self, _dashboard_html(), "text/html; charset=utf-8")
            return
        if route == "/index.html":
            _send_text(self, _dashboard_html(), "text/html; charset=utf-8")
            return
        if route == "/health":
            _json_response(self, 200, {"status": "ok"})
            return
        if route == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if route == "/api/sessions":
            query = parse_qs(parsed.query)
            try:
                limit = int(query.get("limit", ["50"])[0] or 50)
            except (TypeError, ValueError):
                limit = 50
            if limit <= 0:
                limit = 50
            sessions = list_sessions(limit=limit)
            _json_response(self, 200, {"items": sessions})
            return
        if route.startswith("/api/session/"):
            session_id = unquote(route.split("/", 2)[2])
            session = get_session(session_id)
            if session is None:
                _json_response(self, 404, {"error": f"Session {session_id} not found"})
                return
            _json_response(self, 200, session)
            return
        if route.startswith("/static/"):
            if route in ("/static/app.js", "/static/appjs"):
                _send_text(self, _embedded_static("/static/app.js"), "application/javascript; charset=utf-8")
                return
            if route == "/static/styles.css":
                _send_text(self, _embedded_static("/static/styles.css"), "text/css; charset=utf-8")
                return
            _serve_path(route.lstrip("/"), self)
            return

        if route == "/app.js":
            _send_text(self, _embedded_static("/static/app.js"), "application/javascript; charset=utf-8")
            return
        if route == "/appjs":
            _send_text(self, _embedded_static("/static/app.js"), "application/javascript; charset=utf-8")
            return
        if route == "/styles.css":
            _send_text(self, _embedded_static("/static/styles.css"), "text/css; charset=utf-8")
            return

        handler_path = route.lstrip("/")
        candidate = ROOT / handler_path
        if candidate.is_file() and str(candidate).startswith(str(ROOT)):
            if route.endswith(".css"):
                _serve_path(route.lstrip("/"), self)
                return

        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") != "/api/session/finish":
            self.send_error(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            _json_response(self, 400, {"error": "missing_json_body"})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            _json_response(self, 400, {"error": "invalid_json", "detail": str(exc)})
            return

        try:
            session_id = save_session(payload)
            _json_response(self, 200, {"status": "ok", "session_id": session_id})
        except Exception as exc:
            _json_response(self, 500, {"error": "save_failed", "detail": str(exc)})


def main():
    server = ThreadingHTTPServer((HOST, PORT), HGNRequestHandler)
    print(f"HGN local API server listening on http://{HOST}:{PORT}")
    print(f"Loaded handler from: {__file__}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
