# mask_obb_images.py
# ------------------
# 1) ROOT_DIR 아래의 *.json-*.bmp 쌍을 찾는다
# 2) 각 BOX(OBB)에 대해
#       · 장축 방향 EDGE_BAND_LONG px,
#       · 단축 방향 EDGE_BAND_SHORT px 내부 테두리를 살리고
#       · BOX 장축 KEEP_RATIO_W, 단축 KEEP_RATIO_H 중앙부를 살린 뒤
#       · 나머지는 0(검정)으로 마스킹
# 3) 마스킹된 BMP + 원본 JSON을 ROOT_DIR/../post_Mask/ 에 저장
#    (원본 파일은 그대로 보존)

import glob
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------- 사용자 설정 ----------
ROOT_DIR          = r"E:\Experiment_Dataset\03. Richskorea\Post_mask_img\25.04.09"

EDGE_BAND_LONG    = 13           # 장축(long axis) 테두리 폭(px)
EDGE_BAND_SHORT   = 12           # 단축(short axis) 테두리 폭(px)

KEEP_RATIO_W      = 1.0          # 장축 대비 중앙 keep 폭  (1.0 == 100 %)
KEEP_RATIO_H      = 0.40         # 단축 대비 중앙 keep 높이(40 %)

JSON_EXT, IMG_EXT = ".json", ".bmp"
# ---------------------------------

DEST_DIR = Path(ROOT_DIR).parent / "post_Mask"
DEST_DIR.mkdir(exist_ok=True)

# ---------- logging ----------
LOG_FILE = DEST_DIR / "mask_obb.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("maskOBB")
logger.info("출력 폴더 : %s", DEST_DIR.resolve())

# ---------- 유틸 ----------
def list_pairs(root: str, jext=".json", iext=".bmp"):
    """이름이 같은 json-bmp 파일 쌍 목록 반환"""
    pairs = []
    for j in glob.glob(str(Path(root, f"*{jext}"))):
        stem = Path(j).stem
        img_fp = Path(root, f"{stem}{iext}")
        if img_fp.exists():
            pairs.append((j, str(img_fp)))
    return pairs


def axis_vectors(angle_deg: float):
    """angle(°)에 대한 width · height 축 단위 벡터 반환"""
    rad = np.deg2rad(angle_deg)
    u = np.array([np.cos(rad), np.sin(rad)])       # width 방향
    v = np.array([-np.sin(rad), np.cos(rad)])      # height 방향 (u ⟂ v)
    return u, v


def make_rect_poly(center, long_vec, short_vec, long_len, short_len):
    """중심·축벡터·길이로 폴리곤(4점 int32) 생성"""
    c = np.array(center, dtype=np.float32)
    lv = long_vec  * (long_len  / 2.0)
    sv = short_vec * (short_len / 2.0)
    pts = np.array([
        c + lv + sv,
        c + lv - sv,
        c - lv - sv,
        c - lv + sv
    ], dtype=np.int32)
    return pts


def process_image(json_fp: str, img_fp: str):
    stem_name = Path(img_fp).name
    logger.info("▶ 처리 시작: %s", stem_name)

    with open(json_fp, encoding="utf-8") as f:
        data = json.load(f)

    img = cv2.imread(img_fp, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("   이미지 읽기 실패: %s", img_fp)
        return

    h, w = img.shape[:2]
    mask_to_zero = np.zeros((h, w), dtype=np.uint8)   # 누적 마스킹 영역

    
    for shape in data.get("shapes", []):
        if shape.get("label") != "BOX":
            continue

        pts = np.array(shape["points"], dtype=np.float32)
        (cx, cy), (bw, bh), angle = cv2.minAreaRect(pts)

        # --- 장축·단축 결정 ---
        long_len, short_len = (bw, bh) if bw >= bh else (bh, bw)
        u, v = axis_vectors(angle)                       # width(u) · height(v)
        long_vec, short_vec = (u, v) if bw >= bh else (v, u)

        # --- 전체 BOX 마스크 ---
        poly_box = pts.astype(np.int32)
        mask_box = np.zeros_like(mask_to_zero)
        cv2.fillPoly(mask_box, [poly_box], 255)

        # --- keep 영역 (중앙) ---
        keep_w = long_len  * KEEP_RATIO_W
        keep_h = short_len * KEEP_RATIO_H
        poly_keep = make_rect_poly((cx, cy), long_vec, short_vec,
                                   keep_w, keep_h)
        mask_keep = np.zeros_like(mask_to_zero)
        cv2.fillPoly(mask_keep, [poly_keep], 255)

        # --- inner(테두리 제외) ---
        inner_w = max(long_len  - 2 * EDGE_BAND_LONG, 0)
        inner_h = max(short_len - 2 * EDGE_BAND_SHORT, 0)
        if inner_w > 0 and inner_h > 0:
            poly_inner = make_rect_poly((cx, cy), long_vec, short_vec,
                                        inner_w, inner_h)
            mask_inner = np.zeros_like(mask_to_zero)
            cv2.fillPoly(mask_inner, [poly_inner], 255)
            mask_edge = cv2.bitwise_and(mask_box,
                                        cv2.bitwise_not(mask_inner))
        else:
            mask_edge = mask_box.copy()

        mask_allowed = cv2.bitwise_or(mask_keep, mask_edge)

        # --- 누적 마스킹 to_zero ---
        mask_to_zero = cv2.bitwise_or(
            mask_to_zero,
            cv2.bitwise_and(mask_box, cv2.bitwise_not(mask_allowed))
        )

    img_masked = img.copy()
    img_masked[mask_to_zero == 255] = 0

    out_img  = DEST_DIR / Path(img_fp).name
    out_json = DEST_DIR / Path(json_fp).name

    cv2.imwrite(str(out_img), img_masked)
    shutil.copy(json_fp, out_json)

    logger.info("   ⤷ 저장 완료: %s, %s", out_img.name, out_json.name)

if __name__ == "__main__":
    file_pairs = list_pairs(ROOT_DIR, JSON_EXT, IMG_EXT)
    if not file_pairs:
        logger.warning("※ 처리할 파일이 없습니다.")
    else:
        for j_fp, i_fp in tqdm(file_pairs, desc="전체 파일 진행"):
            process_image(j_fp, i_fp)

    logger.info("=== 작업 종료 되었습니다. ===")