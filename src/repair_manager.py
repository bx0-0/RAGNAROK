"""Explicit, user-driven model runtime repair.

Repairs a model by creating an Ollama *derived* model that reuses the same
underlying weights (`FROM <original>`) but applies a user-supplied runtime
configuration (`RENDERER` / `PARSER`). RAGNAROK never guesses the
renderer/parser and never repairs automatically — the user supplies the
configuration via POST /v1/models/repair.

Guarantees (failure-safe order):
    create derived -> verify derived -> switch internal mapping -> delete original
    * creation failure      : original kept, mapping unchanged
    * verification failure  : original kept, derived optionally cleaned up
    * mapping update failure: original kept (delete is only reached after mapping)

Concurrency: one asyncio lock per (model) — the process is single-worker, so an
in-process lock is the correct boundary. A concurrent repair of the same model
serializes; a repeat of an already-applied configuration is detected and
returns the existing derived model without recreating it.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import ollama

from src.logging import logger
from src.model_manager import ModelManager

# Suffix that identifies a RAGNAROK-derived repaired model.
REPAIRED_SUFFIX = "ragnarok-repaired"


# ── model-name helpers ───────────────────────────────────────────────────────
def split_model(name: str) -> Tuple[str, Optional[str]]:
    """Split an Ollama model name into (base, tag).

    `qwen3.8:27b`       -> (`qwen3.8`, `27b`)
    `Qwen...:IQ3_S`     -> (`Qwen...`, `IQ3_S`)
    `llama3.1`          -> (`llama3.1`, None)

    Only the FIRST `:` separates base from tag; the tag may itself contain
    further characters (e.g. uppercase `IQ3_S`). This mirrors Ollama's own
    `model.ParseName` semantics.
    """
    name = name.strip()
    if ":" in name:
        base, _, tag = name.partition(":")
        return base, (tag or None)
    return name, None


def make_repaired_name(original: str, renderer: Optional[str], parser: Optional[str]) -> str:
    """Deterministic, valid Ollama name for a derived model.

    The tag (if any) is preserved as the tag; the base gets a RAGNAROK suffix.
    A short hash of (renderer, parser) disambiguates *different* runtime
    configurations on the same base so two distinct repairs never collide, while
    identical configurations produce the *same* name (idempotent).

    e.g. `Qwen...:IQ3_S` + (r=p) -> `Qwen...-ragnarok-repaired-h1a2b3c4:IQ3_S`
    """
    base, tag = split_model(original)
    basis = f"{renderer or ''}|{parser or ''}"
    h = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:8]
    repaired_base = f"{base}-{REPAIRED_SUFFIX}-{h}"
    return f"{repaired_base}:{tag}" if tag else repaired_base


def parse_modelfile(modelfile: str) -> Dict[str, Optional[str]]:
    """Extract FROM / RENDERER / PARSER from a raw Modelfile.

    Used to reconstruct the repair mapping from Ollama's own stored metadata on
    startup (Ollama persists the Modelfile it created a model from). Returns
    None for keys that are absent. Case-insensitive keywords; the value keeps
    whatever follows the keyword (name may include a tag).
    """
    out = {"from": None, "renderer": None, "parser": None}
    for line in (modelfile or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        kw = parts[0].upper()
        val = parts[1].strip() if len(parts) > 1 else None
        if kw == "FROM" and val:
            out["from"] = val
        elif kw == "RENDERER" and val:
            out["renderer"] = val
        elif kw == "PARSER" and val:
            out["parser"] = val
    return out


def _cfg_key(renderer: Optional[str], parser: Optional[str]) -> str:
    return f"{renderer or ''}|{parser or ''}"


@dataclass
class _RepairRecord:
    """Tracks the repaired model + config applied for a given original name."""
    derived_name: str
    renderer: Optional[str]
    parser: Optional[str]
    cfg_key: str = field(default="")


class RepairManager:
    """Owns repair state and the failure-safe repair sequence."""

    def __init__(self, models: ModelManager):
        self._models = models
        # original requested name -> applied _RepairRecord (the "active mapping")
        self._mapping: Dict[str, _RepairRecord] = {}
        # one lock per original model name (single-process concurrency boundary)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    # ── routing (read on every chat/completion) ─────────────────
    def resolve(self, requested: str) -> str:
        """Map a client-requested model to the model actually sent to Ollama.

        If a repaired model is active for `requested`, return that; otherwise
        return `requested` unchanged.
        """
        rec = self._mapping.get(requested)
        return rec.derived_name if rec else requested

    def active_repair(self, requested: str) -> Optional[_RepairRecord]:
        return self._mapping.get(requested)

    async def reconstruct_from_ollama(self) -> int:
        """Rebuild the in-memory repair mapping from Ollama's own metadata.

        Called at startup. Ollama persists the Modelfile it created each model
        from; a RAGNAROK-derived repaired model's Modelfile is of the form
        `FROM <original>` (+ the RENDERER/PARSER applied). So by listing the
        `<base>-ragnarok-repaired-<hash>` models and reading their modelfile we
        can recover the original-name -> repaired-name mapping that a prior
        process had established (its original may already have been deleted).

        This is what prevents the failure where a successful repair deleted the
        original, but a restart lost the in-memory mapping and made the repaired
        model unreachable. No separate storage or database is introduced.

        Returns the number of mappings restored. Missing Ollama / unreadable
        models are skipped (logged), never fatal.
        """
        restored = 0
        try:
            names = await self._models.list()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"repair reconstruct: Ollama unavailable, skipping: {e}")
            return 0

        for name in names:
            base, _tag = split_model(name)
            if REPAIRED_SUFFIX not in base:
                continue
            try:
                info = await self._models.show(name)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"repair reconstruct: show({name}) failed: {e}")
                continue
            mf = info.get("modelfile") if isinstance(info, dict) else getattr(info, "modelfile", None)
            if not mf:
                logger.warning(f"repair reconstruct: {name} has no modelfile; skipping")
                continue
            parsed = parse_modelfile(mf)
            original = parsed.get("from")
            if not original:
                logger.warning(f"repair reconstruct: {name} modelfile has no FROM; skipping")
                continue
            renderer = parsed.get("renderer")
            parser = parsed.get("parser")
            if original in self._mapping:
                # An in-memory mapping already exists for this original
                # (e.g. a live repair this session); don't clobber it.
                continue
            self._mapping[original] = _RepairRecord(
                derived_name=name,
                renderer=renderer,
                parser=parser,
                cfg_key=_cfg_key(renderer, parser),
            )
            restored += 1
            logger.info(f"repair reconstruct: {original} -> {name}")
        return restored

    def _lock_for(self, model: str) -> asyncio.Lock:
        # Guarded by the outer async context; the per-model dict is only mutated
        # inside _lock_for which is awaited.
        lock = self._locks.get(model)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[model] = lock
        return lock

    async def _locked(self, model: str):
        async with self._locks_guard:
            lock = self._lock_for(model)
        return lock

    # ── modelfile ────────────────────────────────────────────────
    @staticmethod
    def build_modelfile(original: str, renderer: Optional[str], parser: Optional[str]) -> str:
        """Generate the Modelfile from validated user input (no hard-coding)."""
        lines = [f"FROM {original}"]
        if renderer:
            lines.append(f"RENDERER {renderer}")
        if parser:
            lines.append(f"PARSER {parser}")
        return "\n".join(lines) + "\n"

    # ── core repair ──────────────────────────────────────────────
    async def repair(
        self,
        model: str,
        renderer: Optional[str],
        parser: Optional[str],
    ) -> dict:
        """Run the full, failure-safe repair for `model`.

        Returns a result dict (see route layer for the HTTP response shape).
        Raises `RepairError` on any failure — original model is left intact.
        """
        cfg_key = _cfg_key(renderer, parser)
        derived_name = make_repaired_name(model, renderer, parser)

        lock = await self._locked(model)
        async with lock:
            # Idempotency: an identical config already applied -> no work.
            existing = self._mapping.get(model)
            if existing and existing.cfg_key == cfg_key:
                if await self._models.exists(derived_name):
                    logger.info(f"repair({model}) already applied — reuse {derived_name}")
                    return self._result(model, derived_name, renderer, parser, created=False, deleted=False)
                # mapping points at a derived model that no longer exists -> fall through

            if not (renderer or parser):
                # Nothing to apply — not a valid repair.
                raise RepairError(
                    "repair requires at least one of 'renderer' or 'parser'",
                    code="INVALID_PARAMS",
                )

            # 1) derive model must be built FROM an existing original.
            if not await self._models.exists(model):
                raise RepairError(
                    f"original model '{model}' not found in Ollama",
                    code="ORIGINAL_NOT_FOUND",
                )

            # 2) create the derived model (reuse weights via FROM).
            modelfile = self.build_modelfile(model, renderer, parser)
            try:
                await self._models.create_from_modelfile(derived_name, modelfile)
            except ollama.ResponseError as e:
                raise RepairError(
                    f"failed to create derived model: {e.error}", code="CREATE_FAILED",
                ) from e
            except Exception as e:  # connection etc.
                raise RepairError(
                    f"failed to create derived model: {e}", code="CREATE_FAILED",
                ) from e

            # 3) verify the derived model is present and usable.
            verified, vmsg = await self._verify(derived_name)
            if not verified:
                # Optional cleanup of the broken derived model; never touch original.
                await self._safe_delete(derived_name)
                raise RepairError(
                    f"derived model verification failed: {vmsg}", code="VERIFY_FAILED",
                )

            # 4) switch the internal mapping (in-memory, atomic under the lock).
            self._mapping[model] = _RepairRecord(
                derived_name=derived_name,
                renderer=renderer,
                parser=parser,
                cfg_key=cfg_key,
            )
            logger.info(f"repair({model}) mapping -> {derived_name}")

            # 5) only now delete the original (Ollama manages storage).
            deleted = await self._safe_delete(model)

            return self._result(
                model, derived_name, renderer, parser,
                created=True, deleted=deleted,
            )

    async def _verify(self, derived_name: str) -> Tuple[bool, str]:
        """Confirm the derived model exists and can actually serve a chat request.

        Uses a minimal chat probe (system + user, 1 token) so it exercises the
        same Ollama chat-template path the client will hit after repair -- a raw
        generate probe would miss renderer/parser mismatches.
        """
        if not await self._models.exists(derived_name):
            return False, "derived model not present after create"
        try:
            await self._models._client.chat(
                model=derived_name,
                messages=[
                    {"role": "system", "content": "ping"},
                    {"role": "user",   "content": "ping"},
                ],
                options={"num_predict": 1},
                keep_alive="0",
            )
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"chat probe failed: {e}"

    async def _safe_delete(self, name: str) -> bool:
        """Best-effort delete; never raises. Returns True if the delete call succeeded."""
        try:
            await self._models.delete(name)
            logger.info(f"repair: deleted original '{name}'")
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"repair: failed to delete '{name}': {e} (original left in place)")
            return False

    @staticmethod
    def _result(original, derived, renderer, parser, created, deleted) -> dict:
        return {
            "status": "success",
            "original_model": original,
            "repaired_model": derived,
            "renderer": renderer,
            "parser": parser,
            "weights_changed": False,
            "created": created,
            "original_deleted": deleted,
            "active_model": original,  # client keeps using this name
        }


class RepairError(Exception):
    """Repair failed. Carries a machine-readable `code` for the API layer."""

    def __init__(self, message: str, code: str = "REPAIR_FAILED"):
        super().__init__(message)
        self.code = code
