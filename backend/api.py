import asyncio
import gc
import io
import os
import urllib.request
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ป้องกันปัญหาภาพ X-ray ขาดท่อน
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = FastAPI(title="Pes Planus AI API (DenseNet-201 Ensemble)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Pes Planus DenseNet201 API is running perfectly!"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 ชื่อไฟล์โมเดล DenseNet-201 Ensemble (.pt)
MODEL_PATH = "ens_arch_densenet201.pt"

# 🌐 (ตัวเลือก) ใส่ Direct Link สำหรับดาวน์โหลดบน Render กรณีไม่ได้ push ไฟล์เข้า GitHub
# เช่น ลิงก์จาก GitHub Release: "https://github.com/<user>/<repo>/releases/download/v1.0.0/ens_arch_densenet201.pt"
MODEL_DOWNLOAD_URL = os.getenv("MODEL_DOWNLOAD_URL", "")

model_lock = asyncio.Lock()
global_state = {
    "model": None,
    "csv_key": None,
    "gt_map": {},
}

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        if MODEL_DOWNLOAD_URL:
            print(f"⏳ ไม่พบ {MODEL_PATH} กำลังดาวน์โหลดจาก URL...")
            urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, MODEL_PATH)
            print("✅ ดาวน์โหลดโมเดลเรียบร้อยแล้ว!")
        else:
            raise FileNotFoundError(
                f"ไม่พบไฟล์โมเดล '{MODEL_PATH}' ในโฟลเดอร์ และไม่มีการระบุ MODEL_DOWNLOAD_URL"
            )

def load_pt_model():
    download_model_if_needed()
    print("🧠 กำลังโหลด DenseNet-201 Ensemble เข้าสู่หน่วยความจำ...")
    try:
        # ลองโหลดแบบ PyTorch ปกติ
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    except Exception:
        # Fallback กรณีเซฟมาแบบ TorchScript JIT
        model = torch.jit.load(MODEL_PATH, map_location=device)
    
    if isinstance(model, nn.Module):
        model.eval()
    return model

def parse_csv_dataframe(df: pd.DataFrame):
    df.columns = [str(c).strip().replace("\n", "").lower() for c in df.columns]
    img_col = "img_name" if "img_name" in df.columns else None
    label_col = None
    for col in ["label", "label_bin", "patient_label"]:
        if col in df.columns:
            label_col = col
            break

    gt_map = {}
    if img_col and label_col:
        for _, row in df.iterrows():
            if pd.isna(row[img_col]) or pd.isna(row[label_col]):
                continue
            rname = str(row[img_col]).strip().lower()
            b_rname = os.path.splitext(rname)[0]
            raw_lbl = str(row[label_col]).strip().lower()

            if raw_lbl in ["1", "1.0", "flatfoot", "pesplanus", "pes planus", "true"]:
                lbl = 1
            elif raw_lbl in ["0", "0.0", "normal", "false"]:
                lbl = 0
            else:
                try:
                    lbl = int(float(raw_lbl))
                except ValueError:
                    continue

            gt_map[rname] = lbl
            gt_map[b_rname] = lbl
            gt_map[f"{b_rname}.png"] = lbl
            gt_map[f"{b_rname}.jpg"] = lbl
            gt_map[f"{b_rname}.jpeg"] = lbl
    return gt_map

def apply_median_filter(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    median_img = cv2.medianBlur(img_cv, 3)
    return Image.fromarray(cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB))

# DenseNet201 มาตรฐานใช้ขนาด 224x224
def get_transforms():
    return transforms.Compose([
        transforms.Lambda(apply_median_filter),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

@app.post("/api/predict")
async def predict_single_image(
    file: UploadFile = File(...),
    gt_option: str = Form("none"),
    csv_file: Optional[UploadFile] = File(None),
):
    try:
        async with model_lock:
            # 1. โหลดโมเดล DenseNet-201 Ensemble (โหลดเพียงครั้งเดียวตอนเริ่ม)
            if global_state["model"] is None:
                try:
                    global_state["model"] = load_pt_model()
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการโหลดไฟล์โมเดล: {str(e)}")

            # 2. จัดการไฟล์เฉลย CSV (ถ้ามี)
            if gt_option == "upload" and csv_file:
                if global_state["csv_key"] != f"upload_{csv_file.filename}":
                    df_gt = pd.read_csv(io.BytesIO(await csv_file.read()))
                    global_state["gt_map"] = parse_csv_dataframe(df_gt)
                    global_state["csv_key"] = f"upload_{csv_file.filename}"
            else:
                global_state["gt_map"] = {}
                global_state["csv_key"] = "none"

        # 3. เตรียมรูปภาพ
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = get_transforms()(image).unsqueeze(0).to(device)

        # 4. ทำนายผลด้วยโมเดล Ensemble
        model = global_state["model"]
        with torch.no_grad():
            output = model(img_tensor)

            # ตรวจสอบรูปแบบ Output ของโมเดล
            if isinstance(output, tuple):
                output = output[0]

            # แปลง Logits เป็น Probability ด้วย Softmax (หรือ Sigmoid ถ้า output มี node เดียว)
            if output.shape[-1] == 1:
                prob_raw = torch.sigmoid(output).item()
                prediction_result = 1 if prob_raw >= 0.5 else 0
                prob = prob_raw if prediction_result == 1 else (1 - prob_raw)
            else:
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                prediction_result = int(np.argmax(probs))
                prob = float(probs[prediction_result])

        # 5. ตรวจสอบ Ground Truth
        fname = str(file.filename).strip().lower()
        bname = os.path.splitext(fname)[0]
        gt_label = global_state["gt_map"].get(fname) or global_state["gt_map"].get(bname)

        eval_status = "ไม่มีเฉลย"
        if gt_label is not None:
            if gt_label == 1 and prediction_result == 1:
                eval_status = "True Positive (TP)"
            elif gt_label == 0 and prediction_result == 0:
                eval_status = "True Negative (TN)"
            elif gt_label == 0 and prediction_result == 1:
                eval_status = "False Positive (FP)"
            elif gt_label == 1 and prediction_result == 0:
                eval_status = "False Negative (FN)"

        result = {
            "id": file.filename,
            "filename": file.filename,
            "prediction_class": "Pes Planus (ภาวะเท้าแบน)" if prediction_result == 1 else "Normal (ปกติ)",
            "prediction_code": prediction_result,
            "confidence": prob,
            "ground_truth": "Pes Planus (1)" if gt_label == 1 else ("Normal (0)" if gt_label == 0 else "-"),
            "eval_status": eval_status,
        }

        # เคลียร์ Memory ป้องกัน RAM บวม
        del img_tensor, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")