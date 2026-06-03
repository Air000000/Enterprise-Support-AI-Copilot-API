from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]  # 项目根目录

if str(ROOT_DIR) not in sys.path:   # 确保项目根目录在 sys.path 中，这样测试文件才能正确导入项目中的模块
    sys.path.insert(0, str(ROOT_DIR))       