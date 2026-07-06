from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]  # 项目根目录

if str(ROOT_DIR) not in sys.path:   # 确保项目根目录在 sys.path 中，这样测试文件才能正确导入项目中的模块
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture(autouse=True)
def default_auth_user():
    from database import create_db_and_tables
    from auth import CurrentUser, get_current_user
    from main import app

    create_db_and_tables()
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        sub="support",
        user_id="user_demo",
        tenant_id="tenant_demo",
        role="support",
    )
    yield
    app.dependency_overrides.pop(get_current_user, None)
