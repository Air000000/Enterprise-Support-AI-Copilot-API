"""Temporary SQLite only: retain unknown ownership and never silently erase known ownership."""

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import create_engine


@pytest.mark.parametrize("owned", (False, True))
def test_legacy_todo_upgrade_and_guarded_downgrade_preserve_data(tmp_path, monkeypatch, owned):
    url = "sqlite:///" + str(tmp_path / "migration.db")
    monkeypatch.setenv("DATABASE_URL", url)
    config = Config("alembic.ini")
    command.upgrade(config, "20260705_0001")
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("INSERT INTO todo(id,title,completed,due_time) VALUES (7,'old task',0,'tomorrow')")
        command.upgrade(config, "head")
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT id,title,completed,due_time,tenant_id FROM todo").one() == (
                7, "old task", 0, "tomorrow", None)
            assert "ix_todo_tenant_id" in {row[1] for row in connection.exec_driver_sql("PRAGMA index_list(todo)")}
            assert connection.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"
        if owned:
            with engine.begin() as connection:
                connection.exec_driver_sql("UPDATE todo SET tenant_id='tenant_other' WHERE id=7")
            with pytest.raises(RuntimeError, match="Cannot remove Todo tenant ownership"):
                command.downgrade(config, "20260705_0001")
            with engine.connect() as connection:
                assert connection.exec_driver_sql("SELECT tenant_id FROM todo WHERE id=7").scalar_one() == "tenant_other"
                assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == "20261022_0002"
        else:
            command.downgrade(config, "20260705_0001")
            with engine.connect() as connection:
                assert connection.exec_driver_sql("SELECT id,title,due_time FROM todo").one() == (7, "old task", "tomorrow")
            command.upgrade(config, "head")
    finally:
        engine.dispose()
