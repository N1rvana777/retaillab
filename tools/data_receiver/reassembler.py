"""Reassemble split archives and verify checksums.

Takes .part files (or a single .tar.gz) and reconstructs the original archive,
verifying SHA256 integrity at each step.
"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path


class IntegrityError(Exception):
    """Raised when checksum verification fails."""


def reassemble_and_extract(
    source_dir: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, str]]:
    """Reassemble split archive parts and extract contents.

    Args:
        source_dir: Directory containing .part files or a .tar.gz and .sha256 file.
        output_dir: Where to extract the archive contents.

    Returns:
        Tuple of (extraction directory, dict of filename → sha256).

    Raises:
        IntegrityError: If checksum verification fails.
        FileNotFoundError: If required files are missing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find checksum file
    sha_files = list(source_dir.glob("*.sha256"))
    expected_checksums: dict[str, str] = {}
    if sha_files:
        expected_checksums = _parse_checksum_file(sha_files[0])

    # Check if we have parts or a single archive
    parts = sorted(source_dir.glob("*.part"))
    tar_files = list(source_dir.glob("*.tar.gz"))

    if parts:
        # Verify part checksums
        if expected_checksums:
            _verify_checksums(parts, expected_checksums)

        # Reassemble
        archive_path = output_dir / "reassembled.tar.gz"
        _merge_parts(parts, archive_path)
    elif tar_files:
        archive_path = tar_files[0]
        if expected_checksums:
            _verify_checksums([archive_path], expected_checksums)
    else:
        msg = f"在 {source_dir} 中未找到 .part 或 .tar.gz 文件"
        raise FileNotFoundError(msg)

    # Extract
    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir, filter="data")

    # Compute checksums of extracted files
    actual_checksums = {}
    for f in sorted(extract_dir.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(extract_dir))
            actual_checksums[rel] = _sha256(f)

    return extract_dir, actual_checksums


def _parse_checksum_file(path: Path) -> dict[str, str]:
    """Parse a SHA256 checksum file (format: 'hash  filename')."""
    checksums = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                checksums[parts[1].strip()] = parts[0].strip()
    return checksums


def _verify_checksums(
    files: list[Path],
    expected: dict[str, str],
) -> None:
    """Verify SHA256 checksums of files against expected values."""
    for f in files:
        if f.name in expected:
            actual = _sha256(f)
            if actual != expected[f.name]:
                msg = (
                    f"校验失败: {f.name}\n"
                    f"  期望: {expected[f.name]}\n"
                    f"  实际: {actual}"
                )
                raise IntegrityError(msg)


def _merge_parts(parts: list[Path], output: Path) -> None:
    """Merge split .part files back into a single file."""
    with open(output, "wb") as out:
        for part in parts:
            with open(part, "rb") as pf:
                while True:
                    chunk = pf.read(8192)
                    if not chunk:
                        break
                    out.write(chunk)


def _sha256(file_path: Path) -> str:
    """Compute SHA256 hex digest."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
