"""Decision logger — writes every replenishment decision to structured JSONL logs.

The log files are partitioned by date (one file per day) for easy querying
with DuckDB, Pandas, or simple grep. Each line is a self-contained JSON
object representing a single DecisionSnapshot.

Usage:
    logger = DecisionLogger(log_dir=Path("logs/decisions"))
    logger.log(snapshot)
    logger.flush()  # Force write buffered entries
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from quant_engine.telemetry.schemas import DecisionSnapshot

logger = logging.getLogger(__name__)

_DEFAULT_LOG_DIR = Path("logs/decisions")


# 追加写入式的决策日志器：每笔补货决策必须记录，用于事后审计和性能分析
# 注意：非线程安全，多 worker 环境下每个 worker 需要独立的实例
class DecisionLogger:
    """Append-only structured logger for replenishment decisions.

    Thread-safety: NOT thread-safe. Use one logger per worker/process.
    For concurrent API workers, each should have its own logger instance
    writing to a worker-specific file, or use a queue-based approach.
    """

    def __init__(
        self,
        log_dir: Path = _DEFAULT_LOG_DIR,
        buffer_size: int = 100,
    ) -> None:
        """Initialize the decision logger.

        Args:
            log_dir: Directory to write JSONL files to.
            buffer_size: Number of entries to buffer before auto-flushing.
        """
        self._log_dir = log_dir
        # 缓冲区大小：积攒 100 条后一次性写入磁盘，减少 IO 次数
        self._buffer_size = buffer_size
        self._buffer: list[dict] = []
        self._current_date: date | None = None
        self._ensure_dir()

    # 输入：决策快照 → 副作用：缓冲到内存，满了自动刷盘
    def log(self, snapshot: DecisionSnapshot) -> None:
        """Record a single decision.

        Args:
            snapshot: The complete decision audit trail.
        """
        self._buffer.append(snapshot.to_dict())

        # 达到缓冲上限时自动写入磁盘
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    # 输入：决策快照列表 → 副作用：逐条缓冲
    def log_batch(self, snapshots: list[DecisionSnapshot]) -> None:
        """Record multiple decisions at once.

        Args:
            snapshots: List of decision snapshots.
        """
        for snapshot in snapshots:
            self.log(snapshot)

    # 将缓冲区中的所有条目写入磁盘
    def flush(self) -> None:
        """Write all buffered entries to disk."""
        if not self._buffer:
            return

        # 按日期分文件：每天一个文件，方便按日期查询和归档
        today = date.today()
        file_path = self._log_dir / f"decisions_{today.isoformat()}.jsonl"

        try:
            # 追加模式写入，确保不会覆盖已有记录
            with open(file_path, "a", encoding="utf-8") as f:
                for entry in self._buffer:
                    # 每条记录一行 JSON，ensure_ascii=False 支持中文商品名
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            logger.debug(
                "Flushed %d decision entries to %s", len(self._buffer), file_path
            )
        except OSError:
            # 日志写入失败不能影响主流程，只记录告警
            logger.exception("Failed to write decision log to %s", file_path)
        finally:
            # 无论写入是否成功都清空缓冲区，防止内存泄漏
            self._buffer.clear()

    def _ensure_dir(self) -> None:
        """Create the log directory if it doesn't exist."""
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def __del__(self) -> None:
        """Flush remaining buffer on garbage collection."""
        # 对象销毁时尝试刷盘，保证缓冲区中的最后几条记录不丢失
        try:
            self.flush()
        except Exception:
            pass
