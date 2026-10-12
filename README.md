# Thai Restaurant Review Sentiment Analysis

**Five-class (1–5 stars) Thai restaurant-review rating prediction using fine-tuned WangchanBERTa**, with a Flask web interface for interactive and batch inference.

This repository contains the **inference application and demo UI** for the research project published at IEEE InCIT 2025. Data preparation and training notebooks are maintained in the companion repository.

[Published paper](https://doi.org/10.1109/InCIT66780.2025.11276071) · [Live demo](https://sentiment-analysis-gjbh.onrender.com) · [Hugging Face model](https://huggingface.co/iamwarint/sentiment) · [Training notebooks](https://github.com/Aissariya/Sentiment-Analysis-on-Thai-Language-Restaurant-Reviews) · [Presentation](https://www.canva.com/design/DAGjBgfrBL8/SzntuoZ3MaTSYMoGGWFVfQ/view)

## Research overview

The study applies [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased) to classify Thai-language restaurant reviews into five star-rating classes.

| Study detail | Reported value |
| --- | --- |
| Review sources | Wongnai and Google Maps |
| Collected reviews | 7,200 |
| Selected review lengths | 80–510 characters |
| Training split | 6,000 reviews |
| Test split | 1,200 reviews |
| Task | Single-label classification: 1–5 stars |

These study-level details are described in the [published paper abstract](https://doi.org/10.1109/InCIT66780.2025.11276071).

## Published results and metric provenance

| Metric | Value | Source |
| --- | ---: | --- |
| **Fine-tuned WangchanBERTa F1-score** | **80.45%** | Published paper abstract |
| Baseline F1-score | 65.28% | Same published abstract |
| Historical demo accuracy | 80.42% | Previously displayed in this repository; evaluation procedure **not yet verified** |

**F1-score and accuracy are different measures.** The published **80.45% is an F1-score, not accuracy**. The historical **80.42% accuracy** may be a legitimate experimental result, but its exact evaluation run, split, averaging method and checkpoint still need verification. This documentation update did **not** rerun inference or evaluation. The paper-reported F1-score must not be mistaken for a fresh benchmark of the deployed Hugging Face checkpoint.

The companion [fine-tuning notebook](https://github.com/Aissariya/Sentiment-Analysis-on-Thai-Language-Restaurant-Reviews/blob/main/fine_tune_wangchanberta.ipynb) contains calculations for accuracy, F1-score and a classification report, but the published notebook revision does not preserve enough evaluation output to trace the historical 80.42% figure conclusively.

## Features

- Predict a 1–5 star label from one or more Thai restaurant reviews.
- Upload CSV, JSON or TXT for batch predictions.
- View each predicted label and its softmax score.

The softmax score is a model output, **not** a calibrated probability of correctness.

## Quick start

**Requirements:** Python 3.11 is recommended (as in the deployment configuration). An internet connection is needed to download the model from Hugging Face on the first run.

```bash
git clone https://github.com/iamwarint/sentiment-analysis.git
cd sentiment-analysis
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. The first startup can take longer while model weights and tokenizer assets are downloaded.

### Prediction API

```bash
curl -X POST http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{"texts":["อาหารอร่อย บริการดีมาก"]}'
```

The response contains a `predictions` array with a string `label` (1–5) and numeric `score` for each input.

## Project structure

```text
app.py                Flask routes and model inference
templates/index.html  Browser demo (text and file upload)
static/style.css      UI styles
models/final_model/   Saved model/tokenizer artifacts and Git LFS pointer
requirements.txt      Python dependencies
Procfile              Gunicorn entry point
app.yaml              Python 3.11 deployment settings
```

The running app loads model ID `iamwarint/sentiment` from [Hugging Face](https://huggingface.co/iamwarint/sentiment). This repository additionally tracks a local model checkpoint using **Git LFS**; a normal source download may not contain the full LFS weights. Equivalence between these checkpoints has not been established.

## Training and reproducibility

See the companion project's [collection](https://github.com/Aissariya/Sentiment-Analysis-on-Thai-Language-Restaurant-Reviews), [cleaning notebook](https://github.com/Aissariya/Sentiment-Analysis-on-Thai-Language-Restaurant-Reviews/blob/main/data_cleaning.ipynb) and [fine-tuning notebook](https://github.com/Aissariya/Sentiment-Analysis-on-Thai-Language-Restaurant-Reviews/blob/main/fine_tune_wangchanberta.ipynb). This repository focuses on the inference/demo side; it does **not** independently reproduce all figures in the paper.

The dataset originates from third-party review platforms. Check source terms and permissions before redistributing or reusing reviews. **No software license is currently declared in this repository**; public visibility does not imply permission for unrestricted reuse.

## Demo screenshot

![Thai restaurant-review sentiment analysis demo](https://github.com/user-attachments/assets/4e8f59f8-b560-40a1-8470-9cb27eea3e86)

## Publication

**Fine-Tuning Thai Language Model for Sentiment Analysis of Restaurant Reviews Using WangchanBERTa**  
Korawit Orkphol, Kathawach Satianpakiranakorn, Warintorn Jampasee, Aissariya Thongpang, and Jidapa Chaihuadjaroen.  
*2025 9th International Conference on Information Technology (InCIT).*  
DOI: [10.1109/InCIT66780.2025.11276071](https://doi.org/10.1109/InCIT66780.2025.11276071)

## Limitations

The model may make errors on mixed-sentiment reviews, sarcasm, short texts or text outside the training domain. A future reproducibility release should include an exact checkpoint revision, evaluation script, per-class metrics and the test-split manifest.
