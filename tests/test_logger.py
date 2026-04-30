import logging
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.api.dependencies import sanitize_payload_json, sanitize_request_body
from app.utils.logger import (
    BUSINESS_LOG_CHANNEL,
    SuppressSuccessfulEmbeddingsFilter,
    SuppressSuccessfulHTTPXRequestsFilter,
    SYNC_FULL_LOG_CHANNEL,
    SYNC_REBUILD_LOG_CHANNEL,
    configure_logging,
    current_log_channel,
    set_log_channel,
)
from scripts import full_sync as full_sync_script


class LoggerFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = SuppressSuccessfulHTTPXRequestsFilter()

    def test_backward_compatible_filter_alias_points_to_httpx_filter(self):
        self.assertIs(SuppressSuccessfulEmbeddingsFilter, SuppressSuccessfulHTTPXRequestsFilter)

    def test_filter_suppresses_successful_embeddings_requests(self):
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='HTTP Request: POST http://host.docker.internal:9001/v1/embeddings "HTTP/1.1 200 OK"',
            args=(),
            exc_info=None,
        )

        self.assertFalse(self.filter.filter(record))

    def test_filter_keeps_failed_embeddings_requests(self):
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='HTTP Request: POST http://host.docker.internal:9001/v1/embeddings "HTTP/1.1 400 Bad Request"',
            args=(),
            exc_info=None,
        )

        self.assertTrue(self.filter.filter(record))

    def test_filter_suppresses_successful_non_embeddings_requests(self):
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='HTTP Request: DELETE http://qdrant:6333/collections/xinfang_cases "HTTP/1.1 200 OK"',
            args=(),
            exc_info=None,
        )

        self.assertFalse(self.filter.filter(record))

    def test_filter_keeps_failed_non_embeddings_requests(self):
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='HTTP Request: POST http://qdrant:6333/collections/xinfang_cases/points "HTTP/1.1 500 Internal Server Error"',
            args=(),
            exc_info=None,
        )

        self.assertTrue(self.filter.filter(record))

    def test_sanitize_payload_json_redacts_sensitive_fields(self):
        payload_json = sanitize_payload_json(
            {
                "case_id": "CASE-001",
                "description": "案情内容",
                "reported_persons": ["张三"],
                "reporter": "李四",
            }
        )

        self.assertIn('"case_id":"CASE-001"', payload_json)
        self.assertIn('"description":"[REDACTED]"', payload_json)
        self.assertIn('"reported_persons":"[REDACTED]"', payload_json)
        self.assertIn('"reporter":"[REDACTED]"', payload_json)
        self.assertNotIn("案情内容", payload_json)

    def test_sanitize_request_body_redacts_sensitive_fields(self):
        body_json = sanitize_request_body(
            (
                b'{"case_id":"CASE-002","encrypted_description":"secret text",'
                b'"encrypted_reporter":"Alice"}'
            )
        )

        self.assertIn('"case_id":"CASE-002"', body_json)
        self.assertIn('"encrypted_description":"[REDACTED]"', body_json)
        self.assertIn('"encrypted_reporter":"[REDACTED]"', body_json)
        self.assertNotIn("secret text", body_json)

    def test_configure_logging_routes_channels_to_dedicated_files(self):
        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        original_level = root_logger.level
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                log_dir = Path(temp_dir)
                configure_logging("INFO", log_dir=log_dir)
                for handler in root_logger.handlers:
                    if not getattr(handler, "_sx_file_handler", False):
                        handler.setLevel(logging.CRITICAL + 1)
                logger = logging.getLogger("routing-test")

                logger.info("business message", extra={"log_channel": BUSINESS_LOG_CHANNEL})
                logger.info("rebuild message", extra={"log_channel": SYNC_REBUILD_LOG_CHANNEL})
                logger.info("full sync message", extra={"log_channel": SYNC_FULL_LOG_CHANNEL})

                business_log = (log_dir / "business.log").read_text(encoding="utf-8")
                rebuild_log = (log_dir / "sync_rebuild.log").read_text(encoding="utf-8")
                full_sync_log = (log_dir / "sync_full.log").read_text(encoding="utf-8")

                self.assertIn("business message", business_log)
                self.assertIn('"log_channel": "business"', business_log)
                self.assertNotIn("rebuild message", business_log)
                self.assertNotIn("full sync message", business_log)
                self.assertIn("rebuild message", rebuild_log)
                self.assertIn('"log_channel": "sync_rebuild"', rebuild_log)
                self.assertNotIn("business message", rebuild_log)
                self.assertIn("full sync message", full_sync_log)
                self.assertIn('"log_channel": "sync_full"', full_sync_log)
                self.assertNotIn("business message", full_sync_log)
        finally:
            for handler in list(root_logger.handlers):
                root_logger.removeHandler(handler)
                if handler not in original_handlers:
                    handler.close()
            for handler in original_handlers:
                root_logger.addHandler(handler)
            root_logger.setLevel(original_level)


class FullSyncScriptLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_sync_script_uses_sync_full_log_channel(self):
        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        original_level = root_logger.level
        previous_channel = current_log_channel()
        try:
            for handler in list(root_logger.handlers):
                root_logger.removeHandler(handler)

            with tempfile.TemporaryDirectory() as temp_dir:
                log_dir = Path(temp_dir)
                settings = SimpleNamespace(log_level="INFO", log_dir=log_dir)

                async def fake_full_sync(request_id):
                    logging.getLogger("full-sync-script-test").info("script full sync message")
                    return SimpleNamespace(
                        mode="full-sync",
                        total_read=1,
                        total_upserted=1,
                        batches=1,
                        cursor=SimpleNamespace(
                            last_updated_at=datetime.now(timezone.utc),
                            last_case_id="CASE-001",
                        ),
                    )

                container = SimpleNamespace(
                    sync_service=SimpleNamespace(full_sync=fake_full_sync),
                )

                async def fake_build_container(_settings):
                    return container

                async def fake_close_container(_container):
                    return None

                def configure_logging_with_muted_streams(*args, **kwargs):
                    configure_logging(*args, **kwargs)
                    for handler in root_logger.handlers:
                        if not getattr(handler, "_sx_file_handler", False):
                            handler.setLevel(logging.CRITICAL + 1)

                with mock.patch.object(full_sync_script, "get_settings", return_value=settings), mock.patch.object(
                    full_sync_script,
                    "build_container",
                    side_effect=fake_build_container,
                ), mock.patch.object(
                    full_sync_script,
                    "close_container",
                    side_effect=fake_close_container,
                ), mock.patch.object(
                    full_sync_script,
                    "configure_logging",
                    side_effect=configure_logging_with_muted_streams,
                ) as configure_logging_mock, mock.patch("builtins.print"):
                    await full_sync_script.main()

                configure_logging_mock.assert_called_once_with(
                    "INFO",
                    log_dir,
                    default_channel=SYNC_FULL_LOG_CHANNEL,
                )

                business_log = (log_dir / "business.log").read_text(encoding="utf-8")
                rebuild_log = (log_dir / "sync_rebuild.log").read_text(encoding="utf-8")
                full_sync_log = (log_dir / "sync_full.log").read_text(encoding="utf-8")

                self.assertEqual(business_log, "")
                self.assertEqual(rebuild_log, "")
                self.assertIn("script full sync message", full_sync_log)
                self.assertIn('"log_channel": "sync_full"', full_sync_log)
        finally:
            for handler in list(root_logger.handlers):
                root_logger.removeHandler(handler)
                if handler not in original_handlers:
                    handler.close()
            for handler in original_handlers:
                root_logger.addHandler(handler)
            root_logger.setLevel(original_level)
            set_log_channel(previous_channel)
