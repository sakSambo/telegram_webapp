# ASL Telegram Web App

Phone-first Telegram Mini App prototype for the ASL fingerspelling model.

This folder is the only part that needs to be pushed to GitHub for Netlify.
It runs recognition in the browser with:

- MediaPipe Tasks Vision for hand landmarks
- ONNX Runtime Web for local LSTM inference
- `models/asl_lstm_improved_features.onnx`
- `labels/asl_labels_improved_features.json`

The Python backend is optional. Recognition works locally when the browser can load
the ONNX model. Backend APIs are still supported for text assist / sentence assist
or as a fallback.

## Folder Contents

```text
telegram_webapp/
  index.html
  app.js
  local_model.js
  landmark_features.js
  styles.css
  config.js
  netlify.toml
  _headers
  aslguide.jpg
  models/
    asl_lstm_improved_features.onnx
  labels/
    asl_labels_improved_features.json
```

## Refresh Model After Retraining

From the project root:

```powershell
python tools\export_improved_lstm_to_telegram_webapp.py
```

This updates:

```text
telegram_webapp/models/asl_lstm_improved_features.onnx
telegram_webapp/labels/asl_labels_improved_features.json
```

## Local Test

Because the page uses camera permissions and ES modules, serve it over HTTP:

```powershell
python -m http.server 8080 -d telegram_webapp
```

Open:

```text
http://127.0.0.1:8080/
```

For phone testing on the same Wi-Fi network, replace `127.0.0.1` with the computer's
LAN IP address. Some mobile browsers require HTTPS for camera access, so Netlify is
the more realistic test target.

## Netlify Deploy

Option A: small separate GitHub repo.

1. Create a new GitHub repo.
2. Copy only the contents of `telegram_webapp/` into that repo.
3. Connect that repo in Netlify.
4. Build command: leave empty.
5. Publish directory: `.`

Option B: same GitHub repo as the project.

1. Push the whole project only if datasets/videos are excluded by `.gitignore`.
2. In Netlify, set Base directory to `telegram_webapp`.
3. Build command: leave empty.
4. Publish directory: `.`

## Telegram Bot Setup

1. Create a bot with BotFather.
2. Set the Mini App / menu button URL to the Netlify URL.
3. In a chat, open the Web App button.
4. The app receives Telegram theme and init data through `telegram-web-app.js`.

## Optional Backend

If a Python backend is hosted elsewhere, set it in `config.js`:

```javascript
window.ASL_API_BASE = "https://your-backend.example.com";
```

Or enter it from the app's `API` panel.

Backend routes used by optional features:

```text
POST /api/predict_sequence
POST /api/text_assist
POST /api/sentence_assist
GET  /api/health
```

## Notes

- The browser model is the improved 335-feature-per-frame LSTM.
- The raw hand detector still starts from 21 MediaPipe landmarks x 3 coordinates.
- Model assets are cached aggressively by Netlify headers.
- If WebGPU fails, ONNX Runtime falls back to WASM.
