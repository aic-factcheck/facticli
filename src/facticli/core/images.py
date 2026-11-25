"""Claim images: loading, format checks, and the per-run image context.

An image is part of the claim, not evidence: the model sees it so it knows
what is being claimed, and the evidence about it still has to come from
sources with URLs. Images go to the model as base64 data URLs, tagged with a
stable id (``image_1``) so prompts, tool calls, and findings can refer to one.

Nothing is resized. Every accepted format is one the OpenAI API takes as is,
and the provider scales an image to its own token budget; a file above
``MAX_IMAGE_BYTES`` is refused with a message instead of being sent.

Reverse image search needs a URL the search engine can fetch. A URL given as
the image source is used as is; a local file gets one only when the caller
names a public base URL under which the file is mirrored by name.
"""

from __future__ import annotations

import base64
import hashlib
from contextvars import ContextVar, Token
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx
from pydantic import BaseModel, Field

# The per-image limit of the OpenAI API at the time of writing.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30.0

_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


class ImageError(ValueError):
    """An image that cannot be used: missing, unreadable, too large, or not a supported format."""


class ClaimImage(BaseModel):
    """One image that belongs to the claim. Artifacts keep this metadata, never the bytes."""

    image_id: str = Field(description="Stable id the prompts and tools use, e.g. 'image_1'.")
    source: str = Field(description="Path or URL the image was loaded from.")
    mime_type: str
    byte_size: int
    sha256: str
    public_url: str | None = Field(
        default=None,
        description="URL a reverse image search engine can fetch; None when the image has none.",
    )
    data: bytes = Field(default=b"", exclude=True, repr=False)

    @property
    def data_url(self) -> str:
        return f"data:{self.mime_type};base64,{base64.b64encode(self.data).decode('ascii')}"


def sniff_mime_type(data: bytes) -> str | None:
    """MIME type from the file's magic bytes for the formats the model accepts; None otherwise."""
    for signature, mime_type in _SIGNATURES:
        if data.startswith(signature):
            return mime_type
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _is_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")


def public_url_for(path: str, url_base: str | None) -> str | None:
    """URL of a local file mirrored by file name under ``url_base``, or None without a base."""
    if not url_base:
        return None
    return url_base.rstrip("/") + "/" + quote(Path(path).name)


def _download(url: str) -> bytes:
    try:
        with httpx.Client(timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True) as client:
            with client.stream("GET", url, headers={"User-Agent": "facticli"}) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_IMAGE_BYTES:
                        raise ImageError(f"Image at {url} is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
                    chunks.append(chunk)
    except httpx.HTTPError as exc:
        raise ImageError(f"Image could not be downloaded from {url}: {exc}") from exc
    return b"".join(chunks)


def load_image(source: str, *, image_id: str, url_base: str | None = None) -> ClaimImage:
    """Read an image from a local path or an http(s) URL; raise ``ImageError`` when it cannot be used."""
    source = source.strip()
    if not source:
        raise ImageError("Empty image path.")
    if _is_url(source):
        data = _download(source)
        public_url: str | None = source
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise ImageError(f"Image file not found: {source}")
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ImageError(f"Image {source} is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
        data = path.read_bytes()
        public_url = public_url_for(source, url_base)
    mime_type = sniff_mime_type(data)
    if mime_type is None:
        raise ImageError(f"Unsupported image format: {source} (supported: JPEG, PNG, WebP, GIF).")
    return ClaimImage(
        image_id=image_id,
        source=source,
        mime_type=mime_type,
        byte_size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        public_url=public_url,
        data=data,
    )


def load_images(sources: list[str] | tuple[str, ...], *, url_base: str | None = None) -> list[ClaimImage]:
    """Load the claim's images in order, numbered ``image_1``, ``image_2``, ..."""
    return [
        load_image(source, image_id=f"image_{index}", url_base=url_base)
        for index, source in enumerate(sources, start=1)
    ]


# --- per-run context -----------------------------------------------------------

_active_images: ContextVar[tuple[ClaimImage, ...]] = ContextVar("facticli_claim_images", default=())


def activate_images(images: list[ClaimImage] | tuple[ClaimImage, ...]) -> Token:
    """Install the claim's images for the current run, so image tools can resolve ids."""
    return _active_images.set(tuple(images))


def deactivate_images(token: Token) -> None:
    _active_images.reset(token)


def get_images() -> tuple[ClaimImage, ...]:
    """The images of the claim the current run is checking (empty for a text-only claim)."""
    return _active_images.get()


def get_image(image_id: str) -> ClaimImage | None:
    wanted = image_id.strip().lower()
    return next((image for image in _active_images.get() if image.image_id == wanted), None)
