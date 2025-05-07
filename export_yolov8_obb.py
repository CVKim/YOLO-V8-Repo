from ultralytics import YOLO
from pathlib import Path
import shutil, onnx
from onnx import helper, checker

# ── 1.  YOLOv8-OBB  .pt → .onnx  (opset 14)─────────────────────────

## richs 2개 종류 박스 학습한 모델
# WEIGHTS     = r"\\Y\DeepLearning\_projects\rich\25.04.17_BOX_RICHS\outputs\train\weights\best.pt"

## richs 2개 종류 박스 + pre proc 학습한 모델
WEIGHTS     = r"\\Y\DeepLearning\_projects\rich\25.05.01_BOX_RICHS_PREPROC\outputs\train\weights\best.pt"
OUTPUT_DIR  = r"\\Y\DeepLearning\_projects\rich\25.05.01_BOX_RICHS_PREPROC\outputs\train"

model = YOLO(WEIGHTS)
model.export(format="onnx", task="obb", imgsz=(1024, 1024),
             batch=1, opset=14, dynamic=False, device="cuda:0",
             simplify=True, project=OUTPUT_DIR, name="yolov8_obb_static")

export_path = next(Path(OUTPUT_DIR).rglob("*.onnx"))
dest_path   = Path(OUTPUT_DIR) / "yolov8_obb_static.onnx"
shutil.move(export_path, dest_path)

def rename_tensor(model, old, new):
    for tensor in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if tensor.name == old:
            tensor.name = new
    for node in model.graph.node:
        node.input[:]  = [new if x == old else x for x in node.input]
        node.output[:] = [new if x == old else x for x in node.output]

onnx_model = onnx.load(dest_path)

# 일반적으로 images, output0으로 설정 되어 있음
old_in  = onnx_model.graph.input[0].name      
old_out = onnx_model.graph.output[0].name     

rename_tensor(onnx_model, old_in,  "data")
rename_tensor(onnx_model, old_out, "output")

checker.check_model(onnx_model)
onnx.save(onnx_model, dest_path)

print(f"✅ ONNX saved: {dest_path}")
print(f"   └ input  : '{old_in}'  → 'data'")
print(f"   └ output : '{old_out}' → 'output'")