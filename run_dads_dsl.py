#云端python.exe .\run_dads_dsl.py run-config --config configs\serve_cuda.json
#边缘端：python.exe .\run_dads_dsl.py run-config --config configs\client.json   
# 实验：python.exe .\run_dads_dsl.py experiment --config configs\experiment_mobilenet_v2.json
# 实验：python ./run_dads_dsl.py experiment --config configs/experiment_googlenet_block.json

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dads_dsl.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
