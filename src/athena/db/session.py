"""
Database connection session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from athena.core.config import settings

# Create engine instance
engine = create_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class
Base = declarative_base()

# Dependency to get DB session
def get_db():
    """
    Dependency function that yields a SQLAlchemy session.
    To be used in FastAPI dependency injection system.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()