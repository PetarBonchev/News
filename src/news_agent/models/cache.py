from datetime import datetime
from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from news_agent.db.base import Base

class CacheEntry(Base):
    __tablename__ = "cache"

    query_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    articles_json: Mapped[str] = mapped_column(Text, nullable=False)
    raw_response_json: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

