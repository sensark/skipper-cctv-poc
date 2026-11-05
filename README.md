# AI Safety Video Dashboard

A Streamlit dashboard that uploads video to a configured Roboflow workflow and reports detected missing-helmet and missing-safety-vest incidents. Select an incident row to inspect the original and annotated video frame.

## Run locally

Use Python 3.13 for the Roboflow inference SDK. In PowerShell from this folder:

```powershell
py -3.13 -m venv .venv-roboflow
.\.venv-roboflow\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Set `ROBOFLOW_API_KEY` in the environment or enter it in the password field in the sidebar. Do not commit the API key to source control; rotate any key that has been shared publicly. The app uses the configured `sensark` workspace and construction-site workflow. Video inference processes all frames; the confidence threshold is applied to returned predictions. The camera number is included in each incident.

Supported video uploads are MP4, AVI, MOV, MKV, and M4V. OpenCV must be able to decode the video to retrieve incident screenshots.

## Results and interpretation

The workflow must expose a data output named `predictions`, with class labels that identify missing helmets and missing safety vests. Other classes are not reported as PPE incidents. The app does not load local model weights.

Results are model findings for human review, not confirmed safety determinations. Each matching detection is reported per frame; this app does not track or deduplicate people across frames.
