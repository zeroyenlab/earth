# -*- coding: utf-8 -*-
"""★眺める場所。★実験そのものが小さいサーバーを抱える。

★使い方: run.py を走らせて、ブラウザで http://localhost:8765 を開く。
★別サーバーを立てなくていい。★実験を止めれば窓も閉じる。
"""
import json
import os
import threading
import http.server
import socketserver

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("ISLAND_PORT", 8765))

STATE = {"gen": 0, "running": False, "note": "", "conds": {}, "log": [], "cfg": {}}
_LOCK = threading.Lock()


def put(**kw):
    with _LOCK:
        STATE.update(kw)


def log(line, keep=200):
    with _LOCK:
        STATE["log"].insert(0, line)
        del STATE["log"][keep:]


def snap():
    with _LOCK:
        return json.dumps(STATE, ensure_ascii=False)


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/state.json"):
            b = snap().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        try:
            with open(os.path.join(HERE, os.environ.get("ISLAND_VIEW", "view.html")), "rb") as f:
                b = f.read()
        except Exception:
            b = u"<h1>view.html がありません</h1>".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        # ★★HTML にもキャッシュ禁止を付ける（2026-09-10）。
        #   ★付けていなかったので、★**ブラウザが古い画面を掋んだまま**になった。
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class _S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start():
    srv = _S(("127.0.0.1", PORT), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print("★窓を開いた: http://localhost:%d" % PORT, flush=True)
    return srv
