"""Dataset fetchers, tested against a local HTTP server instead of OSF."""

import io
import os
import threading
import warnings
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from whobpyt.datasets import fetchers

FILES = {"/download/aaa": b"first file", "/download/bbb": b"second file"}


def _zip_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("one.txt", "1")
        zf.writestr("sub/two.txt", "2")
    return buf.getvalue()


class _Handler(BaseHTTPRequestHandler):
    hits = []

    def do_GET(self):  # noqa: N802
        self.hits.append(self.path)
        path = self.path.split("?")[0]
        body = _zip_bytes() if path == "/folder.zip" else FILES.get(path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(var, raising=False)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:%d" % httpd.server_address[1]
    monkeypatch.setattr(fetchers, "OSF_DOWNLOAD_URL", base + "/download")
    _Handler.hits = []
    yield base
    httpd.shutdown()


def test_default_data_path():
    assert fetchers.get_localdefaultdatapath() == os.path.expanduser("~/.whobpyt/data")


def test_fetch_osf_files(server, tmp_path):
    dest = tmp_path / "ds"
    out = fetchers._fetch_osf_files({"aaa": "a.txt", "bbb": "nested/b.txt"}, "ds", str(dest))
    assert out == str(dest)
    assert (dest / "a.txt").read_bytes() == b"first file"
    assert (dest / "nested" / "b.txt").read_bytes() == b"second file"


def test_existing_files_are_skipped(server, tmp_path):
    dest = tmp_path / "ds"
    fetchers._fetch_osf_files({"aaa": "a.txt"}, "ds", str(dest))
    n = len(_Handler.hits)
    fetchers._fetch_osf_files({"aaa": "a.txt", "bbb": "b.txt"}, "ds", str(dest))
    assert len(_Handler.hits) == n + 1  # only the missing file


def test_redownload(server, tmp_path):
    dest = tmp_path / "ds"
    fetchers._fetch_osf_files({"aaa": "a.txt"}, "ds", str(dest))
    (dest / "stale.txt").write_text("x")
    fetchers._fetch_osf_files({"aaa": "a.txt"}, "ds", str(dest), redownload=True)
    assert not (dest / "stale.txt").exists()
    assert (dest / "a.txt").exists()


def test_failed_download_leaves_no_file(server, tmp_path):
    dest = tmp_path / "ds"
    with pytest.raises(Exception):
        fetchers._fetch_osf_files({"missing": "m.txt"}, "ds", str(dest))
    assert not (dest / "m.txt").exists()
    assert [p for p in os.listdir(dest) if p.endswith(".part")] == []


def test_fetcher_does_not_change_cwd(server, tmp_path):
    cwd = os.getcwd()
    fetchers._fetch_osf_files({"aaa": "a.txt"}, "ds", str(tmp_path / "ds"))
    assert os.getcwd() == cwd


def test_pull_folder_zip(server, tmp_path):
    out = fetchers.pull_folder(server + "/folder.zip", "myfolder", dest_folder=str(tmp_path))
    assert (tmp_path / "myfolder" / "one.txt").read_text() == "1"
    assert (tmp_path / "myfolder" / "sub" / "two.txt").read_text() == "2"
    assert out == str(tmp_path / "myfolder")
    # Second call is a no-op
    n = len(_Handler.hits)
    fetchers.pull_folder(server + "/folder.zip", "myfolder", dest_folder=str(tmp_path))
    assert len(_Handler.hits) == n


def test_pull_folder_failure_leaves_nothing(server, tmp_path):
    with pytest.raises(Exception):
        fetchers.pull_folder(server + "/nope.zip", "myfolder", dest_folder=str(tmp_path))
    assert os.listdir(tmp_path) == []


def test_bad_download_method(tmp_path):
    with pytest.raises(ValueError):
        fetchers.pull_file("x", str(tmp_path / "x"), download_method="carrier-pigeon")


def test_eggriffiths2022_respects_dest_folder(monkeypatch, tmp_path):
    calls = {}

    def fake_pull_folder(dlcode, newfolder_name, dest_folder=None, download_method="wget"):
        calls.update(newfolder_name=newfolder_name, dest_folder=dest_folder)
        return os.path.join(dest_folder, newfolder_name)

    monkeypatch.setattr(fetchers, "pull_folder", fake_pull_folder)
    out = fetchers.fetch_eggriffiths2022(dest_folder=str(tmp_path))
    assert out == str(tmp_path / "eg__griffiths2022")
    assert calls["dest_folder"] == str(tmp_path)


def test_egismail2025_is_deprecated_alias(monkeypatch):
    monkeypatch.setattr(fetchers, "fetch_egismail2026", lambda **kw: ("2026", kw))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fetchers.fetch_egismail2025(dest_folder="/d", redownload=True)
    assert out == ("2026", {"dest_folder": "/d", "redownload": True})
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


@pytest.mark.parametrize("name, n_files", [
    ("fetch_egtmseeg", 6), ("fetch_egmomi2023", 21),
    ("fetch_egismail2026", 13), ("fetch_egmomi2025", 16),
])
def test_example_fetchers_request_all_files(monkeypatch, tmp_path, name, n_files):
    seen = []
    monkeypatch.setattr(fetchers, "pull_file",
                        lambda url, dest, download_method="wget": seen.append((url, dest)))
    out = getattr(fetchers, name)(dest_folder=str(tmp_path))
    assert out == str(tmp_path)
    assert len(seen) == n_files
    assert all(url.startswith(fetchers.OSF_DOWNLOAD_URL + "/") for url, _ in seen)
    assert all(dest.startswith(str(tmp_path)) for _, dest in seen)
