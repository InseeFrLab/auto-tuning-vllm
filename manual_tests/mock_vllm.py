"""Minimal streaming OpenAI server, launched through the test-only vLLM shim.

Inspired by GuideLLM's mock server: controlled TTFT/ITL, no model or GPU.
https://github.com/vllm-project/guidellm/tree/main/src/guidellm/mock_server
Records actual requests and lifecycle events for independent assertions.
"""

import argparse
import json
import os
import signal
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host")
    parser.add_argument("--max-num-seqs", type=int, required=True)
    args = parser.parse_args()
    events = Path(os.environ["MOCK_EVENTS"]) / f"{os.getpid()}.jsonl"

    def record(kind, **data):
        with events.open("a") as stream:
            stream.write(
                json.dumps(dict(kind=kind, pid=os.getpid(), time=time.time(), **data))
                + "\n"
            )

    record("start", port=args.port, parameter=args.max_num_seqs)

    def stop(signum, _frame):
        record("stop", signal=signum)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    if os.environ.get("MOCK_FAULT") == "startup" and str(
        args.max_num_seqs
    ) == os.environ.get("MOCK_FAIL_PARAMETER"):
        record("startup_error")
        raise SystemExit(1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            data = (
                {"status": "ok"}
                if self.path == "/health"
                else {"object": "list", "data": [{"id": args.model, "object": "model"}]}
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            record("request", path=self.path, body=body)
            if os.environ.get("MOCK_FAULT") == "benchmark" and str(
                args.max_num_seqs
            ) == os.environ.get("MOCK_FAIL_PARAMETER"):
                self.send_error(500, "Injected inference failure")
                return
            chat = self.path.endswith("chat/completions")
            count = int(body.get("max_tokens", 8))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                time.sleep(0.04)
                for index in range(count):
                    choice = {"index": 0, "finish_reason": None}
                    choice["delta" if chat else "text"] = (
                        {"content": "word1 "} if chat else "word1 "
                    )
                    chunk = {
                        "id": "mock",
                        "object": "chat.completion.chunk"
                        if chat
                        else "text_completion",
                        "created": int(time.time()),
                        "model": args.model,
                        "choices": [choice],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.005)
                choice = {"index": 0, "finish_reason": "length"}
                choice["delta" if chat else "text"] = {} if chat else ""
                chunk["choices"] = [choice]
                chunk["usage"] = {
                    "prompt_tokens": 16,
                    "completion_tokens": count,
                    "total_tokens": 16 + count,
                }
                self.wfile.write(
                    f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                record("disconnected")

    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        record("ready")
        server.serve_forever()


if __name__ == "__main__":
    main()
