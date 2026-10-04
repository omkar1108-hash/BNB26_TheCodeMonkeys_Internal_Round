# TrustLayer Chrome Extension (Manifest V3)

A Manifest V3 Chrome browser extension for **TrustLayer** that allows users to select text on any webpage, right-click, and verify its authenticity and manipulation risk against the local TrustLayer multimodal engine.

---

## Architectural & Design Choices

The extension was built in a single pass following Manifest V3 best practices and the exact API specifications of the TrustLayer backend and Streamlit frontend. Key choices made:

1. **Exact API Endpoint and Field Matching**:
   - Examined `README.md`, `backend/main.py`, `backend/api/bundle.py`, and `frontend/app.py`.
   - Identified that `frontend/app.py` submits text bundles via `POST http://localhost:8000/api/bundle/analyze-upload` using the form field `caption`.
   - Used exactly this endpoint (`http://localhost:8000/api/bundle/analyze-upload`) and field (`caption`) sent as `FormData` from the background service worker. No backend or frontend code was altered.

2. **Network Requests in Background Service Worker**:
   - In accordance with Chrome extension security architecture, all network requests are executed exclusively in `background.js` (service worker), never in content scripts.
   - Declared `"host_permissions": ["http://localhost:8000/*"]` with an explanatory comment in `manifest.json` to allow the background worker to communicate with the local server without altering backend CORS settings.

3. **Storage & Ephemeral State Management**:
   - Manifest V3 background service workers are ephemeral and terminate when idle. State is persisted in `chrome.storage.local` under the `analysisState` key (`status`, `selectedText`, `timestamp`, `data`, `errorMessage`).
   - When the user opens the popup, `popup.js` reads the last stored verification state to render the UI immediately.

4. **User Feedback on Context Menu Click**:
   - When the user clicks "🔍 Check with TrustLayer", the action badge immediately shows `"⏳"` with accent `#D8A2A2`.
   - Upon completion, the badge flashes `"✓"` with accent `#8EA66B` (or `"!"` on network/server error). This provides clear visual confirmation of background execution without requesting intrusive permissions like `notifications`.

5. **Strict WCAG AA Color Palette & Typography**:
   - **Popup Background**: `#FFF9D6` (Contrast ratio vs `#2B2B2B` body text: 13.4:1 — AAA).
   - **Card Background**: `#FFDCDC` (Contrast ratio vs `#2B2B2B` text: 11.2:1 — AAA).
   - **Main Button & Accents**: `#D8A2A2` with `#1F1F1F` text (Contrast ratio: 7.4:1 — AAA).
   - **"What looks fine" Accents**: `#8EA66B` with `#1F1F1F` text (Contrast ratio: 6.2:1 — AA).
   - **Body text**: `#2B2B2B`.
   - **Headings & Badges**: `#1F1F1F`.
   - **Zero white text** is used anywhere on these backgrounds.
   - Dimensions: Fixed width of `360px`, `14px` border radius on cards, containers, badges, and buttons, without heavy shadows.

6. **Centralized UI Copy (`popup-copy.js`)**:
   - All visible copy, headings, status messages, verdict labels, emojis, and fallbacks are defined in `popup-copy.js`.
   - `popup.js` dynamically pulls all text from `POPUP_COPY`.

7. **Pure Backend Findings Grounding (Zero Mock Data)**:
   - "How we got here", "⚠️ What's risky to trust", and "✅ What looks fine" extract directly from the backend's real findings: `ranked_evidence`, `cross_modal_results.contradictions`, `cross_modal_results.metadata_conflicts`, `modality_results`, `abstain_reason`, and `conformal_set`.
   - No mock data is used.

8. **Bundled Image Assets**:
   - Created valid PNG icon assets at 16x16, 48x48, and 128x128 in `/extension/icons/` adhering to the palette.

---

## Not yet verified

- **Automated Verification Completed**:
  - `manifest.json` validated as strict, well-formed JSON.
  - All JavaScript files (`background.js`, `popup.js`, `popup-copy.js`) passed static syntax check with `node -c` (zero errors).
  - Popup data formatting logic tested and verified against all four backend output classes (`authentic`, `manipulated`, `coordinated_synthetic`, `insufficient_evidence`).
  - Backend integration confirmed against FastAPI running on Python 3.14.
- **Not yet verified**:
  - Live interactive rendering of the Chrome OS-level right-click context menu was not executed via a browser automation tool, because no headless browser automation framework (such as Playwright or Puppeteer) is installed in this repository. Follow Part C and Part D below to load and verify the unpacked extension in Chrome.

---

## HOW TO RUN THE WHOLE PROJECT

### Part A: Start the backend (Terminal 1, in the project root)
1. Open PowerShell in the project folder.
2. Run: python -m venv venv
3. Run: .\venv\Scripts\Activate.ps1
   If blocked, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
4. Run: python -m pip install -r backend\requirements.txt
5. Run: python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   Leave this window open. Check http://localhost:8000/docs in a browser.

### Part B: Start the Streamlit frontend (Terminal 2, project root, venv activated)
1. Run: python -m streamlit run frontend/app.py
2. Open http://localhost:8501.

### Part C: Load the extension in Chrome
1. Open chrome://extensions in Chrome.
2. Turn on "Developer mode" (top right).
3. Click "Load unpacked" and choose the /extension folder.
4. Pin the TrustLayer icon from the puzzle-piece menu.

### Part D: Use it
1. Go to any article page, select some text, and right-click.
2. Choose "🔍 Check with TrustLayer".
3. Click the TrustLayer icon to see the result.

### Troubleshooting:
- "Can't reach the server": Part A's window is closed or the backend isn't running.
- Popup says nothing yet: you haven't checked any text. Select text and right-click first.
- Ollama is optional. Without it, parts of the analysis run in a simpler mode.
