from __future__ import annotations
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass(frozen=True)
class CmdResult:
    cmd: List[str]
    returncode: int
    stdout: str
    stderr: str

class LastoolsRunner:
    def __init__(self, bin_dir: Path):
        self.bin_dir = Path(bin_dir)

    def exe(self, name: str) -> Path:
        p = self.bin_dir / name
        if not p.exists():
            raise FileNotFoundError(f"LAStools executable not found: {p}")
        return p

    def run(self, args: List[str], timeout_s: Optional[int] = None) -> CmdResult:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            errors="replace",
        )
        return CmdResult(args, p.returncode, p.stdout, p.stderr)

    def lasinfo_text(self, laz_path: Path) -> CmdResult:
        exe = str(self.exe("lasinfo64.exe"))
        cmd = [exe, "-i", str(laz_path), "-quiet", "-stdout"]
        res = self.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"lasinfo failed for {laz_path}\n{res.stderr}")
        return res
