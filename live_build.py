#!/usr/bin/env python3
"""
Live build server for Jupyter Book with auto-rebuild and browser refresh
"""
import subprocess
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class JupyterBookBuilder(FileSystemEventHandler):
    def __init__(self):
        self.last_build = 0
        self.debounce_seconds = 2
        
    def on_any_event(self, event):
        # Ignore build directory and hidden files
        if '_build' in event.src_path or '/.git/' in event.src_path or '.ipynb_checkpoints' in event.src_path:
            return
        
        # Debounce - don't rebuild too frequently
        current_time = time.time()
        if current_time - self.last_build < self.debounce_seconds:
            return
            
        if event.src_path.endswith(('.ipynb', '.md', '.yml', '.yaml')):
            print(f"\n🔄 Change detected: {event.src_path}")
            self.rebuild()
            self.last_build = current_time
    
    def rebuild(self):
        print("🔨 Building Jupyter Book...")
        result = subprocess.run(
            ['.venv/bin/jupyter-book', 'build', '.', '--quiet'],
            cwd=Path.cwd(),
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Build complete at {time.strftime('%H:%M:%S')}")
        else:
            print(f"❌ Build failed:\n{result.stderr}")

def serve_http(port=8000):
    """Serve the built HTML"""
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory='_build/html', **kwargs)
        
        def log_message(self, format, *args):
            pass  # Suppress HTTP logs
    
    server = HTTPServer(('localhost', port), Handler)
    print(f"📡 Serving at http://localhost:{port}")
    server.serve_forever()

if __name__ == '__main__':
    # Initial build
    builder = JupyterBookBuilder()
    builder.rebuild()
    
    # Start file watcher
    observer = Observer()
    observer.schedule(builder, '.', recursive=True)
    observer.start()
    print(f"\n👀 Watching for changes... (Press Ctrl+C to stop)\n")
    
    # Start HTTP server in background
    server_thread = threading.Thread(target=serve_http, daemon=True)
    server_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n\n👋 Stopped watching")
    
    observer.join()
