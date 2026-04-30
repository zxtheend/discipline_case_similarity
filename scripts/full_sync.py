import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.bootstrap import build_container, close_container
from app.config import get_settings
from app.utils.logger import SYNC_FULL_LOG_CHANNEL, configure_logging


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_dir, default_channel=SYNC_FULL_LOG_CHANNEL)
    container = await build_container(settings)
    request_id = "full-sync-{0}".format(
        datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    )
    try:
        result = await container.sync_service.full_sync(request_id=request_id)
        print(
            json.dumps(
                {
                    "request_id": request_id,
                    "mode": result.mode,
                    "total_read": result.total_read,
                    "total_upserted": result.total_upserted,
                    "batches": result.batches,
                    "last_updated_at": result.cursor.last_updated_at.isoformat()
                    if result.cursor.last_updated_at
                    else None,
                    "last_case_id": result.cursor.last_case_id,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        await close_container(container)


if __name__ == "__main__":
    asyncio.run(main())
