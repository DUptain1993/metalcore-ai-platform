"""Integration wrapper around the MIT-licensed RVC-Project.

RVC (Retrieval-based Voice Conversion) is a full project (HuBERT features +
RMVPE f0 + a trained generator + a FAISS feature index), not a pip one-liner. We
*integrate* it rather than reimplement it: :func:`setup` clones the repo and
downloads pretrained weights; the ``*_cmd`` builders construct the standard
training/inference commands; :func:`train_all` / :func:`infer` run them.

The command builders target the **canonical RVC-Project layout**
(``RVC-Project/Retrieval-based-Voice-Conversion-WebUI``, module scripts under
``infer/modules/train/``). If you use a fork with a different CLI, adjust the
builders or override ``rvc_python`` / ``rvc_repo`` in ``configs/rvc.yaml``.
See ``docs/VOCALS_GUIDE.md``.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List

from rvc_training.config import RVCConfig

# Canonical pretrained assets hosted by the RVC-Project maintainers.
_HF_BASE = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
_ASSETS = {
    "assets/hubert/hubert_base.pt": f"{_HF_BASE}/hubert_base.pt",
    "assets/rmvpe/rmvpe.pt": f"{_HF_BASE}/rmvpe.pt",
    "assets/pretrained_v2/f0G40k.pth": f"{_HF_BASE}/pretrained_v2/f0G40k.pth",
    "assets/pretrained_v2/f0D40k.pth": f"{_HF_BASE}/pretrained_v2/f0D40k.pth",
}
_RVC_REPO_URL = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"


def _sr_tag(sample_rate: int) -> str:
    return {40000: "40k", 48000: "48k", 32000: "32k"}.get(sample_rate, "40k")


def _run(cmd: List[str], cwd: Path, logger: logging.Logger) -> None:
    logger.info("$ (cd %s) %s", cwd, " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.stdout:
        logger.debug(result.stdout[-2000:])
    if result.returncode != 0:
        logger.error("Command failed (%d): %s", result.returncode, result.stderr[-3000:])
        raise RuntimeError(f"RVC step failed with code {result.returncode}")


def setup(cfg: RVCConfig, logger: logging.Logger) -> Path:
    """Clone RVC-Project (if absent) and download pretrained weights.

    Returns the repo path. Idempotent: skips clone/download when already present.
    """
    repo = Path(cfg.rvc_repo)
    if not repo.exists():
        logger.info("Cloning RVC-Project -> %s", repo)
        subprocess.run(["git", "clone", "--depth", "1", _RVC_REPO_URL, str(repo)], check=True)
    else:
        logger.info("RVC repo already present at %s", repo)

    for rel, url in _ASSETS.items():
        dest = repo / rel
        if dest.is_file():
            logger.info("Asset present: %s", rel)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s", rel)
        subprocess.run(["wget", "-q", "-O", str(dest), url], check=True)

    logger.info("RVC setup complete. Install its deps once with:  pip install -r %s", repo / "requirements.txt")
    return repo


# --- Command builders (pure; unit-testable) ----------------------------------

def preprocess_cmd(cfg: RVCConfig, dataset_dir: Path, exp_name: str) -> List[str]:
    """Trainset preprocessing (slicing/normalising into the experiment dir)."""
    sr = int(cfg.prep_sample_rate)
    return [
        cfg.rvc_python,
        "infer/modules/train/preprocess.py",
        str(dataset_dir),
        str(sr),
        "2",            # n_p (processes)
        f"logs/{exp_name}",
        "False",        # noparallel
        "3.0",          # per (slice seconds)
    ]


def extract_f0_cmd(cfg: RVCConfig, exp_name: str) -> List[str]:
    """RMVPE pitch extraction across the experiment set."""
    return [
        cfg.rvc_python,
        "infer/modules/train/extract/extract_f0_rmvpe.py",
        "1",            # total parts
        "0",            # part index
        "0",            # gpu index
        f"logs/{exp_name}",
        "True",         # is_half
    ]


def extract_feature_cmd(cfg: RVCConfig, exp_name: str) -> List[str]:
    """HuBERT feature extraction (version v2)."""
    return [
        cfg.rvc_python,
        "infer/modules/train/extract_feature_print.py",
        "cuda:0",
        "1",            # n_part
        "0",            # i_part
        "0",            # i_gpu
        f"logs/{exp_name}",
        "v2",
    ]


def train_cmd(cfg: RVCConfig, exp_name: str) -> List[str]:
    """Generator training with the pretrained v2 40k models."""
    sr = _sr_tag(cfg.prep_sample_rate)
    return [
        cfg.rvc_python,
        "infer/modules/train/train.py",
        "-e", exp_name,
        "-sr", sr,
        "-f0", "1",
        "-bs", str(cfg.batch_size),
        "-g", "0",
        "-te", str(cfg.epochs),
        "-se", str(cfg.save_every_epoch),
        "-pg", f"assets/pretrained_v2/f0G{sr}.pth",
        "-pd", f"assets/pretrained_v2/f0D{sr}.pth",
        "-l", "0",
        "-c", "1" if cfg.cache_in_gpu else "0",
        "-sw", "1",     # save small final model
        "-v", "v2",
    ]


def train_index_cmd(cfg: RVCConfig, exp_name: str) -> List[str]:
    """Build the FAISS feature index for retrieval at inference time."""
    return [cfg.rvc_python, "infer/modules/train/train_index.py", exp_name, "v2"]


def infer_cmd(
    cfg: RVCConfig,
    model_pth: str,
    index_path: str,
    input_wav: str,
    output_wav: str,
) -> List[str]:
    """Voice-conversion inference via RVC-Project's CLI (tools/infer_cli.py)."""
    return [
        cfg.rvc_python,
        "tools/infer_cli.py",
        "--f0up_key", str(cfg.transpose),
        "--input_path", input_wav,
        "--opt_path", output_wav,
        "--model_name", model_pth,
        "--index_path", index_path,
        "--f0method", cfg.f0_method,
        "--index_rate", str(cfg.index_rate),
        "--protect", str(cfg.protect),
    ]


# --- Orchestration -----------------------------------------------------------

def train_all(
    dataset_dir: Path,
    exp_name: str,
    cfg: RVCConfig,
    logger: logging.Logger,
) -> Path:
    """Run the full RVC training pipeline for one experiment.

    Steps: preprocess -> extract f0 -> extract features -> train -> build index.

    Returns the experiment log directory (``<repo>/logs/<exp_name>``) which holds
    the trained weights and the ``added_*.index`` file.
    """
    repo = Path(cfg.rvc_repo)
    if not repo.exists():
        raise FileNotFoundError(
            f"RVC repo not found at {repo}. Run 'setup' first (rvc_training.cli setup)."
        )
    logger.info("=== RVC training: experiment '%s' ===", exp_name)
    _run(preprocess_cmd(cfg, Path(dataset_dir), exp_name), repo, logger)
    _run(extract_f0_cmd(cfg, exp_name), repo, logger)
    _run(extract_feature_cmd(cfg, exp_name), repo, logger)
    _run(train_cmd(cfg, exp_name), repo, logger)
    _run(train_index_cmd(cfg, exp_name), repo, logger)
    exp_dir = repo / "logs" / exp_name
    logger.info("=== RVC training complete -> %s ===", exp_dir)
    return exp_dir


def infer(
    model_pth: str,
    index_path: str,
    input_wav: str,
    output_wav: str,
    cfg: RVCConfig,
    logger: logging.Logger,
) -> Path:
    """Convert ``input_wav`` to the trained voice, writing ``output_wav``."""
    repo = Path(cfg.rvc_repo)
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    _run(infer_cmd(cfg, model_pth, index_path, input_wav, output_wav), repo, logger)
    if not Path(output_wav).is_file():
        raise RuntimeError(f"RVC inference produced no output at {output_wav}")
    return Path(output_wav)
