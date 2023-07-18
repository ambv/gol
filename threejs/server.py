import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse


class SecureContextHTTPRequestHandler(SimpleHTTPRequestHandler):
    """
    Ensures the required Cross-Origin-Opener-Policy and
    Cross-Origin-Embedder-Policy are sent as headers.

    Required for use of SharedArrayBuffer in the web page.
    """

    def end_headers(self):
        self.send_secure_headers()
        super().end_headers()

    def send_secure_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")


httpd = HTTPServer(("localhost", 8000), SecureContextHTTPRequestHandler)
httpd.serve_forever()
