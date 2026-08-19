"""
Single launcher for the Multimodal RAG application.

Starts:
    1. FastAPI backend
    2. Streamlit frontend

Usage:
    python run.py
"""

import subprocess
import sys
import time
import webbrowser


API_HOST = "127.0.0.1"
API_PORT = 8000

STREAMLIT_HOST = "127.0.0.1"
STREAMLIT_PORT = 8501


def main():
    processes = []

    try:
        print("=" * 60)
        print("        MULTIMODAL RAG APPLICATION")
        print("=" * 60)

        # ---------------------------------------------------------
        # Start FastAPI
        # ---------------------------------------------------------

        print("\nStarting FastAPI backend...")

        api_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.api.main:app",
                "--host",
                API_HOST,
                "--port",
                str(API_PORT),
            ]
        )

        processes.append(api_process)

        print(
            f"FastAPI running at "
            f"http://{API_HOST}:{API_PORT}"
        )

        # ---------------------------------------------------------
        # Give FastAPI a moment to start
        # ---------------------------------------------------------

        time.sleep(2)

        # ---------------------------------------------------------
        # Start Streamlit
        # ---------------------------------------------------------

        print("\nStarting Streamlit frontend...")

        streamlit_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.address",
                STREAMLIT_HOST,
                "--server.port",
                str(STREAMLIT_PORT),
                "--server.headless",
                "true",
            ]
        )

        processes.append(streamlit_process)

        streamlit_url = (
            f"http://{STREAMLIT_HOST}:{STREAMLIT_PORT}"
        )

        print(
            f"Streamlit running at {streamlit_url}"
        )

        # ---------------------------------------------------------
        # Open browser
        # ---------------------------------------------------------

        time.sleep(3)

        print("\nOpening Multimodal RAG in your browser...")

        webbrowser.open(streamlit_url)

        # ---------------------------------------------------------
        # Keep both processes alive
        # ---------------------------------------------------------

        print("\n" + "=" * 60)
        print("Multimodal RAG is running.")
        print(f"Frontend : {streamlit_url}")
        print(f"Backend  : http://{API_HOST}:{API_PORT}")
        print("=" * 60)
        print("\nPress Ctrl+C to stop the application.\n")

        while True:

            # If FastAPI crashes, stop everything.
            if api_process.poll() is not None:
                print("FastAPI process stopped.")
                break

            # If Streamlit crashes, stop everything.
            if streamlit_process.poll() is not None:
                print("Streamlit process stopped.")
                break

            time.sleep(1)

    except KeyboardInterrupt:

        print("\n\nStopping Multimodal RAG...")

    finally:

        # ---------------------------------------------------------
        # Stop child processes
        # ---------------------------------------------------------

        for process in processes:

            if process.poll() is None:

                print(
                    f"Stopping process {process.pid}..."
                )

                process.terminate()

        # ---------------------------------------------------------
        # Wait for graceful shutdown
        # ---------------------------------------------------------

        for process in processes:

            try:
                process.wait(timeout=5)

            except subprocess.TimeoutExpired:

                process.kill()

        print("Multimodal RAG stopped.")


if __name__ == "__main__":
    main()