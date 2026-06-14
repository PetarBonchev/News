from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from news_agent.db.base import Base

class CacheEntry(Base):
    __tablename__ = "cache"

    query_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    articles_json: Mapped[str] = mapped_column(Text, nullable=False)
    fetched_at: Mapped[str] = mapped_column(Text, nullable=False)
