/**
 * TrustLayer Extension - Background Service Worker
 * Handles context menu registration, selection handling, API communication with the backend,
 * and storing results in chrome.storage.local.
 */

const BACKEND_URL = "http://localhost:8000/api/bundle/analyze-upload";
const CONTEXT_MENU_ID = "trustlayer-check-selection";

// Register context menu item on install/update
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: CONTEXT_MENU_ID,
      title: "🔍 Check with TrustLayer",
      contexts: ["selection"]
    });
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== CONTEXT_MENU_ID) {
    return;
  }

  const selectedText = (info.selectionText || "").trim();
  if (!selectedText) {
    return;
  }

  // 1. Indicate analysis is in progress via storage and badge
  await chrome.storage.local.set({
    analysisState: {
      status: "analyzing",
      timestamp: Date.now(),
      selectedText: selectedText,
      pageUrl: info.pageUrl || tab?.url || ""
    }
  });

  try {
    await chrome.action.setBadgeText({ text: "⏳" });
    await chrome.action.setBadgeBackgroundColor({ color: "#D8A2A2" });
  } catch (badgeErr) {
    console.warn("Badge update failed:", badgeErr);
  }

  // 2. Perform backend API call
  try {
    const formData = new FormData();
    // Use the exact field 'caption' that frontend/app.py uses for submitting text bundles
    formData.append("caption", selectedText);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 35000);

    const response = await fetch(BACKEND_URL, {
      method: "POST",
      body: formData,
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const result = await response.json();

    // 3. Save successful result in chrome.storage.local
    await chrome.storage.local.set({
      analysisState: {
        status: "success",
        timestamp: Date.now(),
        selectedText: selectedText,
        pageUrl: info.pageUrl || tab?.url || "",
        data: result
      }
    });

    // Update badge to signal completion
    try {
      await chrome.action.setBadgeText({ text: "✓" });
      await chrome.action.setBadgeBackgroundColor({ color: "#8EA66B" });
    } catch (badgeErr) {
      console.warn("Badge update failed:", badgeErr);
    }
  } catch (err) {
    console.error("TrustLayer verification request failed:", err);

    // 4. Save error state in chrome.storage.local
    await chrome.storage.local.set({
      analysisState: {
        status: "error",
        timestamp: Date.now(),
        selectedText: selectedText,
        errorMessage: "📡 TrustLayer can't reach the server. Make sure the backend is running."
      }
    });

    try {
      await chrome.action.setBadgeText({ text: "!" });
      await chrome.action.setBadgeBackgroundColor({ color: "#D8A2A2" });
    } catch (badgeErr) {
      console.warn("Badge update failed:", badgeErr);
    }
  }
});
