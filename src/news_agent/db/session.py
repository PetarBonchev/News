from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from news_agent.config import DATABASE_URL

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)
