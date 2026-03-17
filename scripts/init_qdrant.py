import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.bootstrap import build_container, close_container
from app.config import get_settings
from app.utils.logger import configure_logging


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    container = await build_container(settings)
    try:
        await container.qdrant_service.ensure_collection()
        print("Qdrant collection ready: {0}".format(settings.qdrant_collection))
    finally:
        await close_container(container)


if __name__ == "__main__":
    asyncio.run(main())
