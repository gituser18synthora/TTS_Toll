"""SQLite engine/session management.

The API and the worker are separate processes sharing one SQLite file, so:
  * WAL journal mode (readers never block the writer),
  * a 30 s busy timeout,
  * two engines: read sessions use a deferred BEGIN, write sessions use
    BEGIN IMMEDIATE so a read-then-write transaction can never fail with
    SQLITE_BUSY_SNAPSHOT when the other process committed in between.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

log = logging.getLogger(__name__)

_ro_factory: sessionmaker | None = None
_rw_factory: sessionmaker | None = None
_engines: list[Engine] = []


def _make_engine(db_file: Path, immediate: bool) -> Engine:
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False, "timeout": 30},
        pool_pre_ping=False,
    )

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _record):  # pragma: no cover - trivial
        dbapi_conn.isolation_level = None  # let SQLAlchemy's "begin" event emit BEGIN
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    @event.listens_for(engine, "begin")
    def _on_begin(conn):  # pragma: no cover - trivial
        conn.exec_driver_sql("BEGIN IMMEDIATE" if immediate else "BEGIN")

    return engine


def init_engine(db_file: Path) -> None:
    global _ro_factory, _rw_factory
    dispose()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    ro, rw = _make_engine(db_file, immediate=False), _make_engine(db_file, immediate=True)
    _engines.extend([ro, rw])
    _ro_factory = sessionmaker(bind=ro, expire_on_commit=False)
    _rw_factory = sessionmaker(bind=rw, expire_on_commit=False)


def dispose() -> None:
    global _ro_factory, _rw_factory
    for e in _engines:
        e.dispose()
    _engines.clear()
    _ro_factory = _rw_factory = None


def _factory(write: bool) -> sessionmaker:
    factory = _rw_factory if write else _ro_factory
    if factory is None:
        raise RuntimeError("database not initialised; call init_engine() first")
    return factory


def create_schema() -> None:
    """Create missing tables, then add any columns that newer code expects."""
    engine = _factory(write=True).kw["bind"]
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    _relax_not_null(engine)


def _relax_not_null(engine: Engine) -> None:
    """Rebuild tables whose existing NOT NULL columns became nullable in the models (SQLite can't ALTER that).

    Follows SQLite's documented procedure: new table -> copy rows -> drop -> rename -> indexes,
    with foreign keys off, in one transaction.
    """
    from sqlalchemy import MetaData
    from sqlalchemy.schema import CreateIndex, CreateTable

    for table in Base.metadata.sorted_tables:
        with engine.connect() as conn:
            info = conn.exec_driver_sql(f'PRAGMA table_info("{table.name}")').fetchall()
            conn.rollback()
        notnull = {row[1] for row in info if row[3]}
        relax = [c.name for c in table.columns if c.nullable and not c.primary_key and c.name in notnull]
        if not relax:
            continue
        scratch = MetaData()  # holds the tables the copy's foreign keys point at
        for other in Base.metadata.sorted_tables:
            if other is not table:
                other.to_metadata(scratch)
        tmp = table.to_metadata(scratch, name=f"{table.name}__rebuild")
        cols = ", ".join(f'"{c}"' for c in (row[1] for row in info) if c in table.columns)
        raw = engine.raw_connection()
        try:
            db = raw.driver_connection
            db.execute("PRAGMA foreign_keys=OFF")
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(str(CreateTable(tmp).compile(engine)))
                db.execute(f'INSERT INTO "{tmp.name}" ({cols}) SELECT {cols} FROM "{table.name}"')
                db.execute(f'DROP TABLE "{table.name}"')
                db.execute(f'ALTER TABLE "{tmp.name}" RENAME TO "{table.name}"')
                for idx in table.indexes:
                    db.execute(str(CreateIndex(idx).compile(engine)))
                bad = db.execute("PRAGMA foreign_key_check").fetchall()
                if bad:
                    raise RuntimeError(f"foreign key check failed after rebuilding {table.name}: {bad[:5]}")
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            finally:
                db.execute("PRAGMA foreign_keys=ON")
        finally:
            raw.close()
        log.info("migrated: %s columns %s are now nullable", table.name, relax)


def _add_missing_columns(engine: Engine) -> None:
    with engine.begin() as conn:
        insp = inspect(conn)  # same connection: a second one would wait on our own write lock
        for table in Base.metadata.sorted_tables:
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing:
                    continue
                coltype = col.type.compile(dialect=engine.dialect)
                default = ""
                if col.default is not None and getattr(col.default, "is_scalar", False):
                    val = col.default.arg
                    default = f" DEFAULT {int(val) if isinstance(val, bool) else repr(val)}"
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {coltype}{default}'))
                log.info("migrated: added column %s.%s", table.name, col.name)


@contextmanager
def session_scope(write: bool = False) -> Iterator[Session]:
    """Transactional scope for scripts / the worker. Commits on success."""
    session = _factory(write)()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: read-mostly session."""
    session = _factory(write=False)()
    try:
        yield session
    finally:
        session.close()


def get_db_rw() -> Iterator[Session]:
    """FastAPI dependency: session whose transactions start with BEGIN IMMEDIATE."""
    session = _factory(write=True)()
    try:
        yield session
    finally:
        session.close()
