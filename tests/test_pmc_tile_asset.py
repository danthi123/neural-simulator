from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from tools import pmc_tile_asset as asset


MANIFEST_URL = (
    "https://www.ncbi.nlm.nih.gov/corecgi/tileshop/tileshop.fcgi?"
    "manifest=1&p=PMC3&id=example.jpg&w=1200&h=900"
)
MANIFEST = b'''{
ProjectName:"PMC3",
ImageName:"example.jpg",
Header:"",
Footer:"",
Sat:"28",
aView:[
{sId:"100", sName:"100%", fScale:1, W:5, H:3, w:2, h:2},
{sId:"101", sName:"fit", fScale:0.5, W:3, H:2, w:3, h:2}
]
}
'''


def _png(color: tuple[int, int, int], size: tuple[int, int] = (2, 2)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


class OfflineFetcher:
    def __init__(self, manifest: bytes = MANIFEST) -> None:
        self.manifest = manifest
        self.calls: list[str] = []
        self.tiles: dict[tuple[int, int], bytes] = {
            (1, 1): _png((255, 0, 0)),
            (1, 2): _png((0, 255, 0)),
            (1, 3): _png((0, 0, 255)),
            (2, 1): _png((255, 255, 0)),
            (2, 2): _png((255, 0, 255)),
            (2, 3): _png((0, 255, 255)),
        }

    def __call__(self, url: str) -> asset.FetchResponse:
        self.calls.append(url)
        if url == MANIFEST_URL:
            return asset.FetchResponse(self.manifest, "text/plain; charset=UTF-8", url)
        query = dict(item.split("=", 1) for item in url.split("?", 1)[1].split("&"))
        body = self.tiles[(int(query["r"]), int(query["c"]))]
        return asset.FetchResponse(body, "image/png", url)


def test_acquires_exact_grid_crops_padding_and_writes_self_digested_receipt(tmp_path: Path) -> None:
    fetcher = OfflineFetcher()
    output = tmp_path / "figure.png"
    receipt_path = tmp_path / "figure.receipt.json"

    receipt = asset.acquire(MANIFEST_URL, "100", output, receipt_path, fetcher=fetcher)

    assert fetcher.calls[0] == MANIFEST_URL
    assert fetcher.calls[-1] == MANIFEST_URL
    assert len(fetcher.calls) == 8
    assert receipt["grid"] == {
        "column_count": 3,
        "coordinate_base": 1,
        "row_count": 2,
        "tile_count": 6,
    }
    assert [(tile["row"], tile["column"]) for tile in receipt["tiles"]] == [
        (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)
    ]
    assert len({tile["url"] for tile in receipt["tiles"]}) == 6
    assert receipt["manifest"]["sha256"] == hashlib.sha256(MANIFEST).hexdigest()
    assert receipt["scientific_verdict"] is None
    assert receipt["sha256"] == asset._self_digest(receipt)
    assert asset.verify_receipt(receipt_path) == receipt
    assert hashlib.sha256(output.read_bytes()).hexdigest() == receipt["assembled_image"]["sha256"]

    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (5, 3)
        assert image.getpixel((0, 0)) == (255, 0, 0)
        assert image.getpixel((2, 0)) == (0, 255, 0)
        assert image.getpixel((4, 0)) == (0, 0, 255)
        assert image.getpixel((0, 2)) == (255, 255, 0)
        assert image.getpixel((4, 2)) == (0, 255, 255)


def test_selects_only_the_exact_declared_view(tmp_path: Path) -> None:
    fetcher = OfflineFetcher()
    fetcher.tiles = {(1, 1): _png((1, 2, 3), (3, 2))}

    receipt = asset.acquire(
        MANIFEST_URL,
        "101",
        tmp_path / "fit.png",
        tmp_path / "fit.json",
        fetcher=fetcher,
    )

    assert receipt["view"]["sId"] == "101"
    assert receipt["grid"]["tile_count"] == 1


def test_accepts_exactly_trimmed_edge_tiles(tmp_path: Path) -> None:
    fetcher = OfflineFetcher()
    fetcher.tiles[(1, 3)] = _png((0, 0, 255), (1, 2))
    fetcher.tiles[(2, 1)] = _png((255, 255, 0), (2, 1))
    fetcher.tiles[(2, 2)] = _png((255, 0, 255), (2, 1))
    fetcher.tiles[(2, 3)] = _png((0, 255, 255), (1, 1))

    receipt = asset.acquire(
        MANIFEST_URL,
        "100",
        tmp_path / "trimmed.png",
        tmp_path / "trimmed.json",
        fetcher=fetcher,
    )

    assert [(tile["width"], tile["height"]) for tile in receipt["tiles"]] == [
        (2, 2), (2, 2), (1, 2), (2, 1), (2, 1), (1, 1)
    ]
    with Image.open(tmp_path / "trimmed.png") as image:
        assert image.size == (5, 3)
        assert image.getpixel((4, 2)) == (0, 255, 255)


@pytest.mark.parametrize(
    "manifest, message",
    [
        (MANIFEST.replace(b'ProjectName:"PMC3",', b'ProjectName:"PMC3",ProjectName:"PMC3",'), "duplicate key"),
        (MANIFEST.replace(b'ImageName:"example.jpg"', b'ImageName:"../example.jpg"'), "unsafe"),
        (MANIFEST.replace(b'sId:"101"', b'sId:"100"'), "duplicate view"),
        (MANIFEST + b'__import__("os")', "trailing"),
        (MANIFEST.replace(b'fScale:1', b'fScale:(1)'), "invalid"),
        (MANIFEST.replace(b'W:5', b'W:true'), "invalid"),
    ],
)
def test_rejects_malformed_duplicate_and_executable_manifests(manifest: bytes, message: str) -> None:
    with pytest.raises(asset.PmcTileAssetError, match=message):
        asset.parse_manifest(manifest.decode())


@pytest.mark.parametrize(
    "url",
    [
        MANIFEST_URL.replace("www.ncbi.nlm.nih.gov", "example.org"),
        MANIFEST_URL.replace("id=example.jpg", "id=../example.jpg"),
        MANIFEST_URL + "&p=OTHER",
        MANIFEST_URL.replace("https://", "http://"),
    ],
)
def test_rejects_unofficial_traversing_or_duplicate_manifest_urls(tmp_path: Path, url: str) -> None:
    with pytest.raises(asset.PmcTileAssetError):
        asset.acquire(url, "100", tmp_path / "x.png", tmp_path / "x.json", fetcher=OfflineFetcher())


def test_rejects_missing_wrong_type_wrong_dimensions_and_redirected_tiles(tmp_path: Path) -> None:
    base = OfflineFetcher()

    def missing(url: str) -> asset.FetchResponse:
        if "&r=1&c=2" in url:
            raise OSError("missing")
        return base(url)

    with pytest.raises(asset.PmcTileAssetError, match="fetch failed"):
        asset.acquire(MANIFEST_URL, "100", tmp_path / "a.png", tmp_path / "a.json", fetcher=missing)

    base = OfflineFetcher()

    def wrong_type(url: str) -> asset.FetchResponse:
        response = base(url)
        if "&r=1&c=2" in url:
            return asset.FetchResponse(response.body, "text/html", url)
        return response

    with pytest.raises(asset.PmcTileAssetError, match="content type"):
        asset.acquire(MANIFEST_URL, "100", tmp_path / "b.png", tmp_path / "b.json", fetcher=wrong_type)

    base = OfflineFetcher()
    base.tiles[(1, 2)] = _png((0, 0, 0), (1, 2))
    with pytest.raises(asset.PmcTileAssetError, match="dimensions"):
        asset.acquire(MANIFEST_URL, "100", tmp_path / "c.png", tmp_path / "c.json", fetcher=base)

    base = OfflineFetcher()

    def redirected(url: str) -> asset.FetchResponse:
        response = base(url)
        if "&r=1&c=2" in url:
            return asset.FetchResponse(response.body, response.content_type, url + "&changed=1")
        return response

    with pytest.raises(asset.PmcTileAssetError, match="redirected"):
        asset.acquire(MANIFEST_URL, "100", tmp_path / "d.png", tmp_path / "d.json", fetcher=redirected)


def test_retries_transient_fetch_content_type_failure(tmp_path: Path) -> None:
    base = OfflineFetcher()
    attempts = 0

    def transient(url: str) -> asset.FetchResponse:
        nonlocal attempts
        if "&r=1&c=2" in url:
            attempts += 1
            if attempts == 1:
                return asset.FetchResponse(b"temporary", "text/plain", url)
        return base(url)

    receipt = asset.acquire(
        MANIFEST_URL,
        "100",
        tmp_path / "retry.png",
        tmp_path / "retry.json",
        fetcher=transient,
    )

    assert attempts == 2
    assert receipt["status"] == "acquired"


def test_prior_receipt_pins_manifest_tiles_and_assembled_image(tmp_path: Path) -> None:
    first = asset.acquire(
        MANIFEST_URL,
        "100",
        tmp_path / "first.png",
        tmp_path / "first.json",
        fetcher=OfflineFetcher(),
    )
    changed = OfflineFetcher()
    changed.tiles[(2, 3)] = _png((10, 20, 30))

    with pytest.raises(asset.PmcTileAssetError, match="tile 2,3 changed"):
        asset.acquire(
            MANIFEST_URL,
            "100",
            tmp_path / "second.png",
            tmp_path / "second.json",
            fetcher=changed,
            expected_receipt=first,
        )


def test_rejects_manifest_change_during_download(tmp_path: Path) -> None:
    base = OfflineFetcher()
    manifest_calls = 0

    def changing(url: str) -> asset.FetchResponse:
        nonlocal manifest_calls
        response = base(url)
        if url == MANIFEST_URL:
            manifest_calls += 1
            if manifest_calls == 2:
                return asset.FetchResponse(MANIFEST + b" ", response.content_type, url)
        return response

    with pytest.raises(asset.PmcTileAssetError, match="changed during acquisition"):
        asset.acquire(MANIFEST_URL, "100", tmp_path / "x.png", tmp_path / "x.json", fetcher=changing)

    assert not (tmp_path / "x.png").exists()
    assert not (tmp_path / "x.json").exists()


def test_outputs_are_create_only_and_receipt_tampering_is_detected(tmp_path: Path) -> None:
    output = tmp_path / "figure.png"
    receipt_path = tmp_path / "receipt.json"
    output.write_bytes(b"occupied")
    with pytest.raises(asset.PmcTileAssetError, match="overwrite"):
        asset.acquire(MANIFEST_URL, "100", output, receipt_path, fetcher=OfflineFetcher())
    assert output.read_bytes() == b"occupied"

    output.unlink()
    receipt_path.write_text("occupied")
    with pytest.raises(asset.PmcTileAssetError, match="overwrite"):
        asset.acquire(MANIFEST_URL, "100", output, receipt_path, fetcher=OfflineFetcher())
    assert not output.exists()

    receipt_path.unlink()
    asset.acquire(MANIFEST_URL, "100", output, receipt_path, fetcher=OfflineFetcher())
    value = json.loads(receipt_path.read_text())
    value["tiles"][0]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(value))
    with pytest.raises(asset.PmcTileAssetError, match="self digest"):
        asset.verify_receipt(receipt_path)


def test_receipt_verification_detects_assembled_image_tampering(tmp_path: Path) -> None:
    output = tmp_path / "figure.png"
    receipt_path = tmp_path / "receipt.json"
    asset.acquire(MANIFEST_URL, "100", output, receipt_path, fetcher=OfflineFetcher())

    output.write_bytes(output.read_bytes() + b"tampered")
    with pytest.raises(asset.PmcTileAssetError, match="digest mismatch"):
        asset.verify_receipt(receipt_path)
