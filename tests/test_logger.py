import logging
import unittest

from app.utils.logger import SuppressSuccessfulEmbeddingsFilter


class LoggerFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = SuppressSuccessfulEmbeddingsFilter()

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
