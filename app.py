from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import json

app = Flask(__name__)

# โหลดโมเดล WangchanBERTa
model_path = "./models/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ฟังก์ชันพยากรณ์ข้อความ
def predict_texts(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = [
            {"label": str(torch.argmax(score).item() + 1), "score": float(score.max().item())}
            for score in scores
        ]
    return predictions

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    texts = data.get("texts", [])

    if isinstance(texts, str):  # ถ้ามีข้อความเดียว ให้แปลงเป็นอาร์เรย์
        texts = [texts]

    if not texts:
        return jsonify({"error": "No text provided"}), 400

    results = predict_texts(texts)
    return jsonify({"predictions": results})

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename
    ext = filename.split(".")[-1].lower()

    if ext not in ["csv", "json", "txt"]:
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # อ่านข้อมูลจากไฟล์
        if ext == "csv":
            df = pd.read_csv(file)
            texts = df.iloc[:, 0].dropna().astype(str).tolist()  # ลบ NaN และแปลงเป็น string
        elif ext == "json":
            data = json.load(file)
            texts = [str(item) for item in data if item]  # กรองค่า null ออก
        elif ext == "txt":
            texts = [line.strip() for line in file.read().decode("utf-8").splitlines() if line.strip()]

        if not texts:
            return jsonify({"error": "File is empty or invalid format"}), 400

        # ใช้ predict_texts() เพื่อประมวลผลหลายข้อความพร้อมกัน
        results = predict_texts(texts)

        # ส่งชื่อไฟล์กลับไปด้วย
        return jsonify({"filename": filename, "predictions": results, "texts": texts})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))