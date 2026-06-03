from __future__ import annotations

import glob
import logging
import os
import re
from typing import NamedTuple
from collections.abc import Iterable
from urllib.parse import parse_qs, urlparse

import torch
import yaml
from huggingface_hub import get_token, hf_api, hf_hub_download, snapshot_download
from huggingface_hub.constants import HF_HUB_ENABLE_HF_TRANSFER
from huggingface_hub.utils.tqdm import tqdm as hf_tqdm
from tqdm.contrib.concurrent import thread_map

logger = logging.getLogger(__name__)


_LAYER_FILE_RE = re.compile(r"layer_(\d+)\.safetensors$")


def _discover_layer_safetensors(local_path: str) -> dict[int, str]:
    """Find layer safetensor files, including snapshot symlinks, keyed by layer index."""

    out: dict[int, str] = {}
    for entry in os.scandir(local_path):
        if not entry.name.startswith("layer_") or not entry.name.endswith(".safetensors"):
            continue
        if not (entry.is_file(follow_symlinks=True) or entry.is_symlink()):
            continue
        match = _LAYER_FILE_RE.match(entry.name)
        if not match:
            continue
        out[int(match.group(1))] = entry.path
    return out


def _hf_hub_download_resilient(
    *,
    repo_id: str,
    filename: str,
    revision: str | None = None,
    token: str | None = None,
    force_download: bool = False,
) -> str:
    """Prefer cached local files, then fall back to the network if needed."""

    local_err: Exception | None = None
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
            force_download=force_download,
            local_files_only=True,
        )
    except Exception as e:  # noqa: BLE001
        local_err = e

    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
            force_download=force_download,
        )
    except Exception as e:  # noqa: BLE001
        if local_err is not None:
            logger.warning(
                "HF download fallback failed for %s/%s (local err: %s, network err: %s)",
                repo_id,
                filename,
                local_err,
                e,
            )
        raise


class HfUri(NamedTuple):
    """Structured representation of a HuggingFace URI."""

    repo_id: str
    file_path: str | None
    revision: str | None

    @classmethod
    def from_str(cls, hf_ref: str):
        if hf_ref.startswith("hf://"):
            return parse_hf_uri(hf_ref)

        parts = hf_ref.split("@", 1)
        repo_id = parts[0]
        revision = parts[1] if len(parts) > 1 else None
        return cls(repo_id, None, revision)


def resolve_hf_repo_path(
    hf_ref: str,
    *,
    allow_patterns: list[str] | None = None,
    local_files_only: bool = False,
) -> str:
    """Resolve a repo reference to a local snapshot directory.

    This prefers the local HF cache when ``local_files_only`` is True and avoids
    metadata calls that can fail for gated repos when the snapshot already
    exists on disk.
    """

    hf_uri = HfUri.from_str(hf_ref)
    kwargs = {
        "repo_id": hf_uri.repo_id,
        "revision": hf_uri.revision,
        "local_files_only": local_files_only,
    }
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    return snapshot_download(**kwargs)


def load_transcoder_from_hub(
    hf_ref: str,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = True,
):
    """Load a transcoder from a HuggingFace URI."""

    # resolve legacy references
    if hf_ref == "gemma":
        hf_ref = "mntss/gemma-scope-transcoders"
    elif hf_ref == "llama":
        hf_ref = "mntss/transcoder-Llama-3.2-1B"

    hf_uri = HfUri.from_str(hf_ref)
    try:
        config_path = _hf_hub_download_resilient(
            repo_id=hf_uri.repo_id,
            revision=hf_uri.revision,
            filename="config.yaml",
        )
    except Exception as e:
        raise FileNotFoundError(f"Could not download config.yaml from {hf_uri.repo_id}") from e

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["repo_id"] = hf_uri.repo_id
    config["revision"] = hf_uri.revision
    config["scan"] = f"{hf_uri.repo_id}@{hf_uri.revision}" if hf_uri.revision else hf_uri.repo_id

    return load_transcoders(config, device, dtype, lazy_encoder, lazy_decoder), config


def load_transcoders(
    config: dict,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = True,
):
    """Load a transcoder from a HuggingFace URI."""

    model_kind = config["model_kind"]
    if model_kind == "transcoder_set":
        from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set

        transcoder_paths = resolve_transcoder_paths(config)
        is_gemma_scope = "gemma-scope" in config.get("repo_id", "")

        return load_transcoder_set(
            transcoder_paths,
            scan=config["scan"],
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            gemma_scope=is_gemma_scope,
            dtype=dtype,
            device=device,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
    elif model_kind == "cross_layer_transcoder":
        from circuit_tracer.transcoder.cross_layer_transcoder import load_clt

        try:
            local_path = resolve_hf_repo_path(
                config["repo_id"],
                allow_patterns=["*.safetensors"],
                local_files_only=True,
            )
        except Exception:
            local_path = snapshot_download(
                config["repo_id"],
                revision=config.get("revision", "main"),
                allow_patterns=["*.safetensors"],
            )

        return load_clt(
            local_path,
            scan=config["scan"],
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            lazy_decoder=lazy_decoder,
            lazy_encoder=lazy_encoder,
            dtype=dtype,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")


def resolve_transcoder_paths(config: dict) -> dict:
    if "transcoders" in config:
        hf_paths = [path for path in config["transcoders"] if path.startswith("hf://")]
        local_map = download_hf_uris(hf_paths)
        transcoder_paths = {
            i: local_map.get(path, path) for i, path in enumerate(config["transcoders"])
        }
    else:
        try:
            local_path = resolve_hf_repo_path(
                config["repo_id"],
                allow_patterns=["layer_*.safetensors"],
                local_files_only=True,
            )
        except Exception:
            local_path = snapshot_download(
                config["repo_id"],
                revision=config.get("revision", "main"),
                allow_patterns=["layer_*.safetensors"],
            )
        transcoder_paths = _discover_layer_safetensors(local_path)
        if not transcoder_paths:
            repo_id = config.get("repo_id", "<unknown_repo>")
            revision = config.get("revision", None) or "main"
            raise FileNotFoundError(
                "No layer_*.safetensors were found for transcoder repo "
                f"{repo_id}@{revision}. The repo config loaded, but the layer files "
                f"were not present in the resolved local snapshot: {local_path}. "
                "This usually means the HF cache is incomplete or the model files "
                "have not been downloaded yet."
            )
    return transcoder_paths


def parse_hf_uri(uri: str) -> HfUri:
    """Parse an HF URI into repo id, file path and revision.

    Args:
        uri: String like ``hf://org/repo/file?revision=main``.

    Returns:
        ``HfUri`` with repository id, file path and optional revision.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "hf":
        raise ValueError(f"Not a huggingface URI: {uri}")
    path = parsed.path.lstrip("/")
    repo_parts = path.split("/", 1)
    if len(repo_parts) != 2:
        raise ValueError(f"Invalid huggingface URI: {uri}")
    repo_id = f"{parsed.netloc}/{repo_parts[0]}"
    file_path = repo_parts[1]
    revision = parse_qs(parsed.query).get("revision", [None])[0] or None
    return HfUri(repo_id, file_path, revision)


def download_hf_uri(uri: str) -> str:
    """Download a file referenced by a HuggingFace URI and return the local path."""
    parsed = parse_hf_uri(uri)
    assert parsed.file_path is not None, "File path is not set"
    return _hf_hub_download_resilient(
        repo_id=parsed.repo_id,
        filename=parsed.file_path,
        revision=parsed.revision,
        force_download=False,
    )


def download_hf_uris(uris: Iterable[str], max_workers: int = 8) -> dict[str, str]:
    """Download multiple HuggingFace URIs concurrently with pre-flight auth checks.

    Args:
        uris: Iterable of HF URIs.
        max_workers: Maximum number of parallel workers.

    Returns:
        Mapping from input URI to the local file path on disk.
    """
    if not uris:
        return {}

    uri_list = list(uris)
    if not uri_list:
        return {}
    parsed_map = {uri: parse_hf_uri(uri) for uri in uri_list}

    token = get_token()

    def _download(uri: str) -> str:
        info = parsed_map[uri]
        assert info.file_path is not None, "File path is not set"
        return _hf_hub_download_resilient(
            repo_id=info.repo_id,
            filename=info.file_path,
            revision=info.revision,
            token=token,
            force_download=False,
        )

    # Try a local-cache-only pass first to avoid unnecessary network metadata calls.
    try:
        results = [_download(uri) for uri in uri_list]
        return dict(zip(uri_list, results))
    except Exception as local_or_network_err:  # noqa: BLE001
        logger.info("Local-first HF file resolution did not fully succeed; falling back to repo preflight (%s)", local_or_network_err)

    # ---  Pre-flight Check ---
    logger.info("Performing pre-flight metadata check...")
    unique_repos = {info.repo_id for info in parsed_map.values()}

    for repo_id in unique_repos:
        if hf_api.repo_info(repo_id=repo_id, token=token).gated is not False:
            if token is None:
                raise PermissionError("Cannot access a gated repo without a hf token.")

    logger.info("Pre-flight check complete. Starting downloads...")

    if HF_HUB_ENABLE_HF_TRANSFER:
        # Use a simple loop for sequential download if HF_TRANSFER is enabled
        results = [_download(uri) for uri in uri_list]
        return dict(zip(uri_list, results))

    # The thread_map will attempt all downloads in parallel. If any worker thread
    # raises an exception (like GatedRepoError from _download), thread_map
    # will propagate that first exception, failing the entire process.
    results = thread_map(
        _download,
        uri_list,
        desc=f"Fetching {len(parsed_map)} files",
        max_workers=max_workers,
        tqdm_class=hf_tqdm,
    )
    return dict(zip(uri_list, results))
