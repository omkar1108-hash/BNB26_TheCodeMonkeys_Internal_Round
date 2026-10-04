# TrustLayer Extension - Open Questions & Future Considerations

This document records architectural nuances, uncertainties, and future feature possibilities identified during extension development.

---

### 1. Programmatic Popup Opening vs. Visual Badge Notification
- **Observation**: Chrome's Extension API does not generally permit service workers to open the action popup window autonomously without an immediate user gesture on Chrome desktop (`chrome.action.openPopup` is restricted).
- **Current Approach**: When a right-click check is triggered, the background service worker immediately updates the badge to `"⏳"` (accent `#D8A2A2`) and flips to `"✓"` (accent `#8EA66B`) when the verification completes. The user clicks the pinned extension icon to view the popup.
- **Question**: Would an optional desktop notification (`chrome.notifications`) or in-page toast banner be desired in future revisions to notify users when a long verification completes?

### 2. Multi-Context Right-Click (Images & Links)
- **Observation**: The context menu is currently registered with `contexts: ["selection"]` to analyze textual articles and claims.
- **Current Approach**: The service worker sends the highlighted text to the backend's `/api/bundle/analyze-upload` endpoint under the `caption` field.
- **Question**: Should future versions support right-clicking images on a webpage (`contexts: ["image"]`) to download image bytes and submit them as an image artifact bundle to `/api/bundle/analyze-upload`?

### 3. Deep-Linking to Specific Streamlit Analysis Results
- **Observation**: The "📋 Open full report" button opens `http://localhost:8501`.
- **Current Approach**: The backend assigns a unique `bundle_id` to each analysis session, and the Streamlit app (`frontend/app.py`) is designed as a standalone interactive dashboard without query parameter routing for individual bundle IDs.
- **Question**: If URL query parameter support is added to `frontend/app.py` in the future (e.g. `http://localhost:8501/?bundle_id=<id>`), should the extension pass the saved `bundle_id` to direct the user directly to their specific investigated bundle?
