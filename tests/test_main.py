import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import httpx

import app.main as main_module
from app.bootstrap import close_container
from app.container import ReadinessProbe
from app.models.response import ClueMiningResponse, HealthResponse as HealthResponseModel, IdentifyResponse
from app.main import _resolve_request_log_channel, create_app
from app.utils.logger import (
    BUSINESS_LOG_CHANNEL,
    SYNC_FULL_LOG_CHANNEL,
    SYNC_REBUILD_LOG_CHANNEL,
    configure_logging,
    current_log_channel,
    set_log_channel,
)


class MainAppTests(unittest.IsolatedAsyncioTestCase):
    def test_resolve_request_log_channel_maps_routes_to_expected_files(self):
        api_prefix = "/api/v1"

        self.assertEqual(_resolve_request_log_channel("/health", api_prefix), BUSINESS_LOG_CHANNEL)
        self.assertEqual(_resolve_request_log_channel("/ready", api_prefix), BUSINESS_LOG_CHANNEL)
        self.assertEqual(_resolve_request_log_channel("/ready/sync", api_prefix), BUSINESS_LOG_CHANNEL)
        self.assertEqual(
            _resolve_request_log_channel("/api/v1/identify", api_prefix),
            BUSINESS_LOG_CHANNEL,
        )
        self.assertEqual(
            _resolve_request_log_channel("/api/v1/clues", api_prefix),
            BUSINESS_LOG_CHANNEL,
        )
        self.assertEqual(
            _resolve_request_log_channel("/api/v1/admin/sync/rebuild-row", api_prefix),
            SYNC_REBUILD_LOG_CHANNEL,
        )
        self.assertEqual(
            _resolve_request_log_channel("/api/v1/admin/sync/full", api_prefix),
            SYNC_FULL_LOG_CHANNEL,
        )
        self.assertIsNone(_resolve_request_log_channel("/api/v1/admin/sync/incremental", api_prefix))

    async def test_request_validation_error_logs_sanitized_json_body(self):
        app = create_app()
        app.state.container = SimpleNamespace()
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            with self.assertLogs("main", level="WARNING") as captured:
                response = await client.post(
                    "/api/v1/admin/sync/rebuild-row",
                    headers={"X-Request-ID": "req-422"},
                    json={
                        "case_id": "CASE-100",
                        "source_wtxx_bh": "XFJ-100",
                        "location": "太原市",
                        "encrypted_description": "案情内容",
                    },
                )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["request_id"], "req-422")
        combined_logs = "\n".join(captured.output)
        self.assertIn("request_validation_error", combined_logs)
        self.assertIn("req-422", combined_logs)
        self.assertIn('"case_id":"CASE-100"', combined_logs)
        self.assertIn('"encrypted_description":"[REDACTED]"', combined_logs)
        self.assertNotIn("案情内容", combined_logs)

    async def test_middleware_and_request_context_route_logs_to_expected_files(self):
        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        original_level = root_logger.level
        previous_channel = current_log_channel()
        try:
            for handler in list(root_logger.handlers):
                root_logger.removeHandler(handler)

            with tempfile.TemporaryDirectory() as temp_dir:
                log_dir = Path(temp_dir)
                configure_logging("INFO", log_dir=log_dir)
                for handler in root_logger.handlers:
                    if not getattr(handler, "_sx_file_handler", False):
                        handler.setLevel(logging.CRITICAL + 1)

                app = create_app()
                route_logger = logging.getLogger("request-routing-test")

                class FakePipeline:
                    async def identify(self, payload, request_id):
                        route_logger.info("identify route marker")
                        return IdentifyResponse(
                            similar_cases=[],
                            processing_time_ms=1,
                            request_id=request_id,
                        )

                    async def mine_clues(self, payload, request_id):
                        route_logger.info("clues route marker")
                        return ClueMiningResponse(
                            incremental_clues=[],
                            supplemental_clues=[],
                            processing_time_ms=1,
                            request_id=request_id,
                        )

                class FakeSyncService:
                    async def rebuild_row(self, request_id, row):
                        route_logger.info("rebuild route marker")
                        return SimpleNamespace(
                            started_at=datetime.now(timezone.utc),
                            finished_at=datetime.now(timezone.utc),
                            mode="rebuild-row",
                            total_read=1,
                            total_upserted=1,
                            batches=1,
                            cursor=SimpleNamespace(
                                last_updated_at=None,
                                last_case_id=row.case_id,
                            ),
                        )

                    async def full_sync(self, request_id):
                        route_logger.info("full sync route marker")
                        return SimpleNamespace(
                            started_at=datetime.now(timezone.utc),
                            finished_at=datetime.now(timezone.utc),
                            mode="full-sync",
                            total_read=1,
                            total_upserted=1,
                            batches=1,
                            cursor=SimpleNamespace(
                                last_updated_at=None,
                                last_case_id="CASE-FULL",
                            ),
                        )

                async def ready_probe():
                    route_logger.info("ready route marker")

                app.state.container = SimpleNamespace(
                    pipeline=FakePipeline(),
                    sync_service=FakeSyncService(),
                    get_readiness_probes=lambda name: {
                        "default": (
                            ReadinessProbe(name="qdrant", check=ready_probe),
                        ),
                        "sync": (),
                    }.get(name, ()),
                )

                def logged_health_response(*args, **kwargs):
                    route_logger.info("health route marker")
                    return HealthResponseModel(*args, **kwargs)

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    with mock.patch.object(
                        main_module,
                        "HealthResponse",
                        side_effect=logged_health_response,
                    ):
                        health_response = await client.get("/health")
                        ready_response = await client.get("/ready")
                        identify_response = await client.post(
                            "/api/v1/identify",
                            json={
                                "reported_persons": ["张三"],
                                "reporter": "李四",
                                "location": "太原市",
                                "description": "案情内容",
                            },
                        )
                        clues_response = await client.post(
                            "/api/v1/clues",
                            json={
                                "reported_persons": ["张三"],
                                "reporter": "李四",
                                "location": "太原市",
                                "description": "案情内容",
                                "similar_case": {
                                    "case_id": "CASE-001",
                                    "reported_persons": ["王五"],
                                    "description_text": "历史案情",
                                },
                            },
                        )
                        rebuild_response = await client.post(
                            "/api/v1/admin/sync/rebuild-row",
                            json={
                                "case_id": "CASE-REBUILD",
                                "source_wtxx_bh": "WT-001",
                                "petition_id": "XFJ-001",
                                "create_time": "2026-01-01T00:00:00Z",
                            },
                        )
                        full_sync_response = await client.post("/api/v1/admin/sync/full")

                self.assertEqual(health_response.status_code, 200)
                self.assertEqual(ready_response.status_code, 200)
                self.assertEqual(identify_response.status_code, 200)
                self.assertEqual(clues_response.status_code, 200)
                self.assertEqual(rebuild_response.status_code, 200)
                self.assertEqual(full_sync_response.status_code, 200)

                business_log = (log_dir / "business.log").read_text(encoding="utf-8")
                rebuild_log = (log_dir / "sync_rebuild.log").read_text(encoding="utf-8")
                full_sync_log = (log_dir / "sync_full.log").read_text(encoding="utf-8")

                self.assertIn("health route marker", business_log)
                self.assertIn("ready route marker", business_log)
                self.assertIn("identify route marker", business_log)
                self.assertIn("clues route marker", business_log)
                self.assertNotIn("rebuild route marker", business_log)
                self.assertNotIn("full sync route marker", business_log)
                self.assertIn("rebuild route marker", rebuild_log)
                self.assertNotIn("identify route marker", rebuild_log)
                self.assertNotIn("full sync route marker", rebuild_log)
                self.assertIn("full sync route marker", full_sync_log)
                self.assertNotIn("health route marker", full_sync_log)
                self.assertNotIn("rebuild route marker", full_sync_log)
        finally:
            for handler in list(root_logger.handlers):
                root_logger.removeHandler(handler)
                if handler not in original_handlers:
                    handler.close()
            for handler in original_handlers:
                root_logger.addHandler(handler)
            root_logger.setLevel(original_level)
            set_log_channel(previous_channel)

    async def test_ready_endpoints_use_container_readiness_registry(self):
        app = create_app()

        async def healthy():
            return None

        async def unhealthy():
            raise RuntimeError("embedding offline")

        app.state.container = SimpleNamespace(
            get_readiness_probes=lambda name: {
                "default": (
                    ReadinessProbe(name="qdrant", check=healthy),
                    ReadinessProbe(name="embedding", check=unhealthy),
                ),
                "sync": (
                    ReadinessProbe(name="mysql", check=healthy),
                ),
            }.get(name, ())
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            ready_response = await client.get("/ready")
            sync_response = await client.get("/ready/sync")

        self.assertEqual(ready_response.status_code, 200)
        self.assertEqual(ready_response.json()["status"], "degraded")
        self.assertEqual(
            ready_response.json()["dependencies"],
            [
                {"name": "qdrant", "healthy": True, "detail": None},
                {"name": "embedding", "healthy": False, "detail": "embedding offline"},
            ],
        )
        self.assertEqual(sync_response.status_code, 200)
        self.assertEqual(sync_response.json()["status"], "ok")
        self.assertEqual(
            sync_response.json()["dependencies"],
            [{"name": "mysql", "healthy": True, "detail": None}],
        )

    async def test_close_container_continues_after_individual_shutdown_failures(self):
        closed_components = []

        async def fail_close():
            closed_components.append("failing")
            raise RuntimeError("close failed")

        async def ok_close():
            closed_components.append("healthy")

        container = SimpleNamespace(
            shutdown_callbacks=(
                ("failing", fail_close),
                ("healthy", ok_close),
            )
        )

        with self.assertLogs("bootstrap", level="WARNING") as captured:
            await close_container(container)

        self.assertEqual(closed_components, ["failing", "healthy"])
        self.assertIn("container_shutdown_failed", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
