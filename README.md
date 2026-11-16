# AI Safety Video Dashboard

A Streamlit dashboard that analyzes uploaded video using either a PPE-trained Roboflow workflow or a general-purpose YOLO baseline. Select an incident row to inspect the original and annotated frame.

## Run locally

Use Python 3.13 for the Roboflow inference SDK. In PowerShell from this folder:

```powershell
py -3.13 -m venv .venv-roboflow
.\.venv-roboflow\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Set `ROBOFLOW_API_KEY` in the environment to use the PPE-trained model, or create a project-root `.env` file containing `ROBOFLOW_API_KEY=your_roboflow_api_key`. The app loads `.env` at startup; restart Streamlit after changing the file. `.env` is ignored by Git. The key is not entered in the dashboard. Do not commit or share the API key; rotate any key that has been exposed. The app uses the `sensark` workspace's PPE workflow on `serverless.roboflow.com`, with header-based API-key authentication and cached results. Select **YOLO26n baseline (COCO)** to run a standard COCO-pretrained object detector locally; it detects general object classes, not helmets or safety vests, so PPE alerts and compliance are not reported in baseline mode. Baseline weights are checked and, if missing, downloaded and SHA-256 verified automatically when the app starts. They are stored at `models/yolo26n.pt` and excluded from source control by the existing `*.pt` ignore rule. The detector itself loads into memory when baseline inference is first requested. The confidence threshold applies to either model, and the camera number is included in each incident.

Ultralytics YOLO is distributed under AGPL-3.0. Review the [Ultralytics license terms](https://www.ultralytics.com/license) before redistributing or using the baseline in a proprietary or commercial deployment.

Supported video uploads are MP4, AVI, MOV, MKV, and M4V. OpenCV must be able to decode the video to retrieve incident screenshots.

## Results and interpretation

The trained workflow must expose a data output named `predictions`, with class labels that identify missing helmets and missing safety vests. Other classes are not reported as PPE incidents. The YOLO baseline's person/object counts are detections per analyzed frame, not unique tracked people.

Results are model findings for human review, not confirmed safety determinations. Each matching detection is reported per frame; this app does not track or deduplicate people across frames.
