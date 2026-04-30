from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Tuple

from app.config import Settings
from app.core.pipeline import IdentifyPipeline
from app.sync.data_sync import DataSyncService


ReadinessCheck = Callable[[], Awaitable[None]]
ShutdownCallback = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class ReadinessProbe:
    name: str
    check: ReadinessCheck


@dataclass
class ApplicationContainer:
    settings: Settings
    sync_service: DataSyncService
    pipeline: IdentifyPipeline
    readiness_registry: Dict[str, Tuple[ReadinessProbe, ...]] = field(default_factory=dict, repr=False)
    shutdown_callbacks: Tuple[Tuple[str, ShutdownCallback], ...] = field(
        default_factory=tuple,
        repr=False,
    )

    def get_readiness_probes(self, name: str) -> Tuple[ReadinessProbe, ...]:
        return self.readiness_registry.get(name, ())
