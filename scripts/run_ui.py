import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clown_vision import ui

if __name__ == "__main__":
    ui.main()