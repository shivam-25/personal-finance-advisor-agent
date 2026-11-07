"""Three memory tiers: conversation (rolling dialogue), working (one turn,
shared between agents) and profile (durable per-customer facts)."""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import settings
from .domain import utc_now
from .supabase_backend import SupabaseBackend, supabase_configured

STOPWORDS = frozenset(
    """a an the my me i is are was were do does did can could should would how what
    why when which to of for in on at and or but with about from that this it""".split()
)


@dataclass
class Turn:
    role: str
    content: str
    at: str = field(default_factory=utc_now)
    intent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConversationMemory:
    def __init__(self, max_turns: int = 12) -> None:
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, role: str, content: str, intent: str | None = None) -> None:
        self._turns.append(Turn(role=role, content=content, intent=intent))

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def last_user_turn(self) -> Turn | None:
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn
        return None

    def last_intent(self) -> str | None:
        for turn in reversed(self._turns):
            if turn.role == "user" and turn.intent:
                return turn.intent
        return None

    def as_messages(self, limit: int = 6) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns[-limit:]]

    def transcript(self, limit: int = 6) -> str:
        return "\n".join(
            f"{t.role}: {t.content}" for t in self.turns[-limit:]
        )


class WorkingMemory:
    """Per-turn blackboard shared by the planner and the specialists."""

    def __init__(self) -> None:
        self._slots: dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        self._slots[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._slots.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._slots

    def keys(self) -> list[str]:
        return sorted(self._slots)

    def clear(self) -> None:
        self._slots.clear()

    def snapshot(self) -> dict[str, Any]:
        return dict(self._slots)


@dataclass
class MemoryNote:
    kind: str
    key: str
    value: Any
    note: str = ""
    at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryBackend:
    """Storage seam: JSON files locally, Supabase where the filesystem is ephemeral."""

    def load(self, customer_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def save(self, customer_id: str, notes: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    @property
    def label(self) -> str:
        return type(self).__name__


class JsonFileBackend(MemoryBackend):
    def __init__(self, directory: Path | None = None) -> None:
        self._dir = Path(directory or settings.memory_dir)

    def _path(self, customer_id: str) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir / f"{customer_id}.json"

    def load(self, customer_id: str) -> list[dict[str, Any]]:
        path = self._path(customer_id)
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("notes", [])
        except (json.JSONDecodeError, OSError):
            return []

    def save(self, customer_id: str, notes: list[dict[str, Any]]) -> None:
        payload = {
            "customer_id": customer_id,
            "updated_at": utc_now(),
            "notes": notes,
        }
        try:
            self._path(customer_id).write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:
            pass  # read-only filesystem: memory stays in-process

    @property
    def label(self) -> str:
        return f"json:{self._dir}"


def default_backend() -> MemoryBackend:
    if supabase_configured():
        try:
            return SupabaseBackend()
        except Exception:
            pass
    return JsonFileBackend()


class ProfileMemory:
    """Durable customer-scoped memory held behind a storage backend."""

    def __init__(
        self,
        customer_id: str,
        backend: MemoryBackend | None = None,
        directory: Path | None = None,
    ) -> None:
        self.customer_id = customer_id
        if backend is None:
            backend = JsonFileBackend(directory) if directory else default_backend()
        self.backend = backend
        self._notes: list[MemoryNote] = []
        self._load()

    def _load(self) -> None:
        rows = self.backend.load(self.customer_id)
        notes = []
        for row in rows:
            try:
                notes.append(MemoryNote(**{k: row[k] for k in ("kind", "key", "value", "note", "at") if k in row}))
            except TypeError:
                continue
        self._notes = notes

    def _save(self) -> None:
        # Memory writes are best-effort: a storage failure must not break a turn.
        try:
            self.backend.save(self.customer_id, [n.to_dict() for n in self._notes])
        except Exception:
            pass

    def remember(self, kind: str, key: str, value: Any, note: str = "") -> MemoryNote:
        """Upsert on (kind, key) so repeated statements do not pile up."""
        entry = MemoryNote(kind=kind, key=key, value=value, note=note)
        self._notes = [n for n in self._notes if not (n.kind == kind and n.key == key)]
        self._notes.append(entry)
        self._save()
        return entry

    def forget(self, kind: str, key: str) -> bool:
        before = len(self._notes)
        self._notes = [n for n in self._notes if not (n.kind == kind and n.key == key)]
        if len(self._notes) != before:
            self._save()
            return True
        return False

    def all(self, kind: str | None = None) -> list[MemoryNote]:
        return [n for n in self._notes if kind is None or n.kind == kind]

    def recall(self, query: str, limit: int = 5) -> list[MemoryNote]:
        """Keyword-overlap retrieval."""
        terms = _tokens(query)
        if not terms:
            return self._notes[-limit:]
        scored = []
        for note in self._notes:
            haystack = _tokens(f"{note.kind} {note.key} {note.value} {note.note}")
            overlap = len(terms & haystack)
            if overlap:
                scored.append((overlap, note))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [note for _, note in scored[:limit]]

    def constraints(self) -> dict[str, Any]:
        """Preferences that must shape generated advice."""
        out: dict[str, Any] = {"protected_categories": [], "declined": [], "goals": []}
        for note in self._notes:
            if note.kind == "preference" and note.value == "protected":
                out["protected_categories"].append(note.key)
            elif note.kind == "declined_recommendation":
                out["declined"].append(note.key)
            elif note.kind == "goal":
                out["goals"].append({"key": note.key, "value": note.value})
        return out

    def snapshot(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "backend": self.backend.label,
            "note_count": len(self._notes),
            "notes": [n.to_dict() for n in self._notes],
            "constraints": self.constraints(),
        }


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9&]+", str(text).lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


class MemoryStore:
    """The three tiers, scoped to one session."""

    def __init__(
        self,
        customer_id: str,
        session_id: str,
        backend: "MemoryBackend | None" = None,
    ) -> None:
        self.customer_id = customer_id
        self.session_id = session_id
        self.conversation = ConversationMemory()
        self.working = WorkingMemory()
        self.profile = ProfileMemory(customer_id, backend=backend)

    def start_turn(self) -> None:
        """Working memory is turn-scoped; the other tiers survive."""
        self.working.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "conversation": [t.to_dict() for t in self.conversation.turns],
            "working": sorted(self.working.keys()),
            "profile": self.profile.snapshot(),
        }
