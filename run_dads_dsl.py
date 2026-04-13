#云端D:\Program4Python\anaconda\envs\pytorch\python.exe .\run_dads_dsl.py serve --host 0.0.0.0 --port 50051 --device cuda
#边缘端：D:\Program4Python\anaconda\envs\pytorch\python.exe .\run_dads_dsl.py client-run --server localhost:50051 --model mobilenet_v2 --bandwidth-mbps 5 --cpu-load-target 0 --report-format table

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dads_dsl.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
