# run_all.py —— 依次运行所有 python 示例，输出分隔清晰
# 用法：python run_all.py  （或 python python/run_all.py）
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
scripts = sorted(HERE.glob("0*.py"))   # 01~06，跳过 run_all.py 自身

for s in scripts:
    print("\n" + "=" * 72, flush=True)
    print(f"  运行 {s.name}", flush=True)
    print("=" * 72, flush=True)
    code = subprocess.call([sys.executable, str(s)])
    if code != 0:
        print(f"[警告] {s.name} 退出码 {code}", flush=True)
