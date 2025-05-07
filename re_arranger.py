"""
Richskorea data re-splitter
───────────────────────────
• SOURCE_ROOT : 올바른 train/val 기준 경로  (index prefix 포함)
• TARGET_ROOT : train/val 구별 없이 파일이 섞여 있는 경로 (prefix 없음)
• DST_ROOT    : 새로 split‧이름을 정리해 넣을 경로            (prefix 포함)
"""

import shutil
from pathlib import Path
from tqdm import tqdm

# ── 사용자 설정 ───────────────────────────────────────────────────────────────
SOURCE_ROOT = Path(r"E:\Experiment_Dataset\03. Richskorea\DataSplit\04.17_BOX")
TARGET_ROOT = Path(r"E:\Experiment_Dataset\03. Richskorea\Post_mask_img\25.04.09_MASK")
DST_ROOT    = Path(r"E:\Experiment_Dataset\03. Richskorea\DataSplit\05.01_BOX")

IMG_EXT, LABEL_EXT = ".bmp", ".json"
# ─────────────────────────────────────────────────────────────────────────────

def source_map(root: Path, split: str) -> dict[str, tuple[str, str]]:
    """
    source/train(or val)에 있는 파일들을
    {본체 key: (index prefix, split)} 형태로 반환
    (중복 key가 생기면 첫 항목 유지)
    """
    m: dict[str, tuple[str, str]] = {}
    for f in (root / split).iterdir():
        if f.suffix.lower() not in {IMG_EXT, LABEL_EXT}:
            continue
        parts = f.stem.split("_", 1)
        if len(parts) != 2:
            continue                      # 예외적 형식 무시
        prefix, key = parts
        m.setdefault(key, (prefix, split))
    return m

# 1) source 전체 매핑: key → (prefix, split)
source_info: dict[str, tuple[str, str]] = {}
for sp in ("train", "val"):
    source_info.update(source_map(SOURCE_ROOT, sp))

# 2) target 내부에서 쌍(bmp+json)이 모두 있는 key만 추리기
bmp_keys   = {f.stem for f in TARGET_ROOT.glob(f"*{IMG_EXT}") if f.is_file()}
json_keys  = {f.stem for f in TARGET_ROOT.glob(f"*{LABEL_EXT}") if f.is_file()}
target_pairs = bmp_keys & json_keys                    # 완전한 쌍

# 3) 재배치
total = len(source_info)
with tqdm(total=total, desc="Relocating") as bar:
    for key, (prefix, split) in source_info.items():
        bar.update()
        if key not in target_pairs:               # 대상 없음
            continue

        # 새 파일명 및 목적지 폴더
        new_name = f"{prefix}_{key}"
        dst_dir  = DST_ROOT / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        for ext in (IMG_EXT, LABEL_EXT):
            src = TARGET_ROOT / f"{key}{ext}"
            dst = dst_dir / f"{new_name}{ext}"

            if not src.exists():
                tqdm.write(f"[누락] {src} (건너뜀)")
                break
            if dst.exists():
                tqdm.write(f"[충돌] {dst} 이미 존재 (건너뜀)")
                break
        else:
            for ext in (IMG_EXT, LABEL_EXT):
                src = TARGET_ROOT / f"{key}{ext}"
                dst = dst_dir / f"{new_name}{ext}"
                shutil.move(src, dst)

print("=== 재배치 완료 ===")
