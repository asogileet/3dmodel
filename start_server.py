import http.server
import webbrowser
import os
import sys
from functools import partial

# Ensure UTF-8 output on Windows consoles
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DEFAULT_PORT = 8080

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and caching headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-cache, must-revalidate')
        super().end_headers()

    def guess_type(self, path):
        if path.endswith('.glb'):
            return 'model/gltf-binary'
        if path.endswith('.gltf'):
            return 'model/gltf+json'
        if path.endswith('.js') or path.endswith('.mjs'):
            return 'application/javascript'
        if path.endswith('.css'):
            return 'text/css'
        return super().guess_type(path)

def run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    handler_factory = partial(CustomHTTPRequestHandler, directory=script_dir)

    chosen_port = None
    httpd = None

    for port in range(DEFAULT_PORT, DEFAULT_PORT + 30):
        try:
            httpd = http.server.ThreadingHTTPServer(('0.0.0.0', port), handler_factory)
            chosen_port = port
            break
        except OSError:
            continue

    if not httpd:
        print("❌ 無法找到可用的連接埠，請檢查是否有過多服務正在執行。")
        return

    url = f"http://localhost:{chosen_port}/index.html"
    print("=" * 64)
    print(" 🚀 [Gloria 3D 骨骼操作平台] 本地伺服器已成功啟動！")
    print(f" 🌐 網頁網址: {url}")
    print(f" 📂 根目錄:   {script_dir}")
    print(" 💡 按下 Ctrl+C 即可隨時停止伺服器")
    print("=" * 64)

    try:
        webbrowser.open(url)
    except Exception as e:
        print("提示: 自動開啟瀏覽器失敗，請手動複製上方網址開啟：", e)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n伺服器已停止。")
    finally:
        httpd.server_close()

if __name__ == '__main__':
    run()
