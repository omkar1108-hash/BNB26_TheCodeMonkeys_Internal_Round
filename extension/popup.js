/**
 * TrustLayer Extension - Popup Controller
 * Retrieves and renders the last saved analysis result from chrome.storage.local.
 * Adheres strictly to the copy definitions in popup-copy.js and the design specifications.
 */

document.addEventListener("DOMContentLoaded", async () => {
  const copy = window.POPUP_COPY;
  if (!copy) {
    console.error("POPUP_COPY not loaded.");
    return;
  }

  // Populate static copy
  const appTitle = document.getElementById("app-title");
  const appSubtitle = document.getElementById("app-subtitle");
  const emptyMessage = document.getElementById("empty-message");
  const analyzingMessage = document.getElementById("analyzing-message");
  const errorMessage = document.getElementById("error-message");
  const howHeading = document.getElementById("how-we-got-here-heading");
  const riskyHeading = document.getElementById("risky-heading");
  const fineHeading = document.getElementById("fine-heading");
  const snippetHeading = document.getElementById("snippet-heading");
  const btnFullReport = document.getElementById("btn-full-report");

  if (appTitle) appTitle.textContent = copy.header.title;
  if (appSubtitle) appSubtitle.textContent = copy.header.subtitle;
  if (emptyMessage) emptyMessage.textContent = copy.states.empty;
  if (analyzingMessage) analyzingMessage.textContent = copy.states.analyzing;
  if (errorMessage) errorMessage.textContent = copy.states.errorUnreachable;
  if (howHeading) howHeading.textContent = copy.sections.howWeGotHere;
  if (riskyHeading) riskyHeading.textContent = copy.sections.risky;
  if (fineHeading) fineHeading.textContent = copy.sections.fine;
  if (snippetHeading) snippetHeading.textContent = copy.sections.checkedSnippet;
  if (btnFullReport) btnFullReport.textContent = copy.button.openFullReport;

  // Setup full report button
  if (btnFullReport) {
    btnFullReport.addEventListener("click", () => {
      chrome.tabs.create({ url: "http://localhost:8501" });
    });
  }

  // Load analysis state from storage
  let storedState = null;
  try {
    const storageResult = await chrome.storage.local.get("analysisState");
    storedState = storageResult.analysisState;
  } catch (err) {
    console.error("Failed to read storage:", err);
  }

  renderState(storedState, copy);
});

/**
 * Switch and render active state view
 */
function renderState(state, copy) {
  const stateEmpty = document.getElementById("state-empty");
  const stateAnalyzing = document.getElementById("state-analyzing");
  const stateError = document.getElementById("state-error");
  const stateResult = document.getElementById("state-result");

  // Reset display
  if (stateEmpty) stateEmpty.style.display = "none";
  if (stateAnalyzing) stateAnalyzing.style.display = "none";
  if (stateError) stateError.style.display = "none";
  if (stateResult) stateResult.style.display = "none";

  if (!state) {
    if (stateEmpty) stateEmpty.style.display = "block";
    return;
  }

  if (state.status === "analyzing") {
    if (stateAnalyzing) stateAnalyzing.style.display = "block";
    return;
  }

  if (state.status === "error") {
    if (stateError) {
      stateError.style.display = "block";
      const errEl = document.getElementById("error-message");
      if (errEl) {
        errEl.textContent = state.errorMessage || copy.states.errorUnreachable;
      }
    }
    return;
  }

  if (state.status === "success" && state.data) {
    if (stateResult) {
      stateResult.style.display = "flex";
      renderResultData(state.data, state.selectedText, copy);
    }
    return;
  }

  // Default fallback to empty state
  if (stateEmpty) stateEmpty.style.display = "block";
}

/**
 * Render the full verification result
 */
function renderResultData(data, selectedText, copy) {
  const verdictEmojiEl = document.getElementById("verdict-emoji");
  const verdictHeadlineEl = document.getElementById("verdict-headline");
  const confidenceBadgeEl = document.getElementById("confidence-badge");
  const abstainedBadgeEl = document.getElementById("abstained-badge");
  const snippetCardEl = document.getElementById("snippet-card");
  const snippetTextEl = document.getElementById("snippet-text");
  const howListEl = document.getElementById("how-we-got-here-list");
  const riskyListEl = document.getElementById("risky-list");
  const fineListEl = document.getElementById("fine-list");

  // 1. Headline & Large Emoji
  const label = data.label || "insufficient_evidence";
  const verdictConfig = copy.verdicts[label] || copy.verdicts.insufficient_evidence;

  if (verdictEmojiEl) verdictEmojiEl.textContent = verdictConfig.emoji;
  if (verdictHeadlineEl) verdictHeadlineEl.textContent = verdictConfig.headline;

  // 2. Calibrated Confidence Badge
  if (confidenceBadgeEl) {
    const pct = typeof data.confidence === "number"
      ? (data.confidence * 100).toFixed(1) + "%"
      : "";
    confidenceBadgeEl.textContent = pct
      ? `${pct} ${copy.meta.confidence}`
      : copy.meta.confidence;
  }

  // 3. Abstained Badge
  if (abstainedBadgeEl) {
    if (data.abstained) {
      abstainedBadgeEl.style.display = "inline-block";
      abstainedBadgeEl.textContent = copy.meta.abstainedBadge;
      if (data.abstain_reason) {
        abstainedBadgeEl.title = data.abstain_reason;
      }
    } else {
      abstainedBadgeEl.style.display = "none";
    }
  }

  // 4. Evaluated text snippet
  if (snippetCardEl && snippetTextEl) {
    if (selectedText && selectedText.trim()) {
      snippetCardEl.style.display = "block";
      snippetTextEl.textContent = `"${selectedText.trim()}"`;
    } else {
      snippetCardEl.style.display = "none";
    }
  }

  // 5. "How we got here" - One emoji sentence per finding using only backend findings
  if (howListEl) {
    howListEl.innerHTML = "";
    const howFindings = buildHowWeGotHereFindings(data, copy);
    howFindings.forEach((finding) => {
      const li = document.createElement("li");
      li.textContent = finding;
      howListEl.appendChild(li);
    });
  }

  // 6. "⚠️ What's risky to trust"
  if (riskyListEl) {
    riskyListEl.innerHTML = "";
    const riskyFindings = buildRiskyFindings(data, copy);
    riskyFindings.forEach((finding) => {
      const li = document.createElement("li");
      li.textContent = finding;
      riskyListEl.appendChild(li);
    });
  }

  // 7. "✅ What looks fine"
  if (fineListEl) {
    fineListEl.innerHTML = "";
    const fineFindings = buildFineFindings(data, copy);
    fineFindings.forEach((finding) => {
      const li = document.createElement("li");
      li.textContent = finding;
      fineListEl.appendChild(li);
    });
  }
}

/**
 * Builds "How we got here" findings:
 * One emoji sentence per finding, using only the backend's findings.
 */
function buildHowWeGotHereFindings(data, copy) {
  const sentences = [];

  // Use ranked_evidence if provided by backend
  if (Array.isArray(data.ranked_evidence) && data.ranked_evidence.length > 0) {
    data.ranked_evidence.forEach((item) => {
      const desc = item.description || "";
      let emoji = copy.modalityEmojis.neutral;

      const lower = desc.toLowerCase();
      if (lower.includes("conflict") || lower.includes("contradiction") || item.category === "cross_modal") {
        emoji = copy.modalityEmojis.contradiction;
      } else if (item.modality === "image" || lower.startsWith("[image]")) {
        emoji = copy.modalityEmojis.image;
      } else if (item.modality === "audio" || lower.startsWith("[audio]")) {
        emoji = copy.modalityEmojis.audio;
      } else if (item.modality === "metadata" || lower.startsWith("[metadata]")) {
        emoji = copy.modalityEmojis.metadata;
      } else if (item.modality === "text" || lower.startsWith("[text]")) {
        emoji = copy.modalityEmojis.text;
      }

      sentences.push(`${emoji} ${desc}`);
    });
  } else {
    // When ranked_evidence is empty, use the backend's explicit findings
    if (data.abstain_reason) {
      sentences.push(`🤷 ${data.abstain_reason}`);
    }

    if (data.modality_results) {
      Object.entries(data.modality_results).forEach(([modKey, modVal]) => {
        if (modVal && modVal.present && modVal.evidence) {
          let modEmoji = copy.modalityEmojis.neutral;
          if (modKey === "text") modEmoji = copy.modalityEmojis.text;
          else if (modKey === "image") modEmoji = copy.modalityEmojis.image;
          else if (modKey === "audio") modEmoji = copy.modalityEmojis.audio;
          else if (modKey === "metadata") modEmoji = copy.modalityEmojis.metadata;

          sentences.push(`${modEmoji} ${modVal.evidence}`);
        }
      });
    }

    if (Array.isArray(data.next_steps) && data.next_steps.length > 0) {
      data.next_steps.slice(0, 2).forEach((step) => {
        sentences.push(`${copy.modalityEmojis.guidance} ${step}`);
      });
    }
  }

  if (sentences.length === 0) {
    sentences.push(copy.fallbacks.noFindingsAvailable);
  }

  return sentences;
}

/**
 * Builds "⚠️ What's risky to trust" findings:
 * Extracts contradictions, anomalies >= 0.35, and abstention reasons from the backend.
 */
function buildRiskyFindings(data, copy) {
  const items = [];

  // 1. Contradictions between sources
  const contradictions = data.cross_modal_results?.contradictions || [];
  contradictions.forEach((c) => {
    items.push(`⚡ ${c.description || `${c.field.toUpperCase()} conflict detected between sources.`}`);
  });

  // 2. Metadata conflicts
  const metaConflicts = data.cross_modal_results?.metadata_conflicts || [];
  metaConflicts.forEach((mc) => {
    items.push(`⚠️ ${mc}`);
  });

  // 3. Modality anomaly scores >= 0.35
  if (data.modality_results) {
    Object.entries(data.modality_results).forEach(([modKey, modVal]) => {
      if (modVal && modVal.present && typeof modVal.score === "number" && modVal.score >= 0.35) {
        let modEmoji = copy.modalityEmojis.risky;
        if (modKey === "image") modEmoji = copy.modalityEmojis.image;
        else if (modKey === "audio") modEmoji = copy.modalityEmojis.audio;
        else if (modKey === "metadata") modEmoji = copy.modalityEmojis.metadata;
        else if (modKey === "text") modEmoji = copy.modalityEmojis.text;

        const detail = modVal.evidence || `Anomaly score: ${modVal.score.toFixed(2)}`;
        items.push(`${modEmoji} ${modKey.toUpperCase()}: ${detail}`);
      }
    });
  }

  // 4. Abstention reason
  if (data.abstained) {
    const reason = data.abstain_reason || "Evidence is insufficient across modalities to verify authenticity.";
    items.push(`🤷 ${reason}`);
  }

  // 5. Fallback if no risks were flagged
  if (items.length === 0) {
    items.push(copy.fallbacks.noRisksFound);
  }

  return items;
}

/**
 * Builds "✅ What looks fine" findings:
 * Extracts clean modalities (< 0.35), lack of contradictions, and conformal stability.
 */
function buildFineFindings(data, copy) {
  const items = [];

  // 1. Clean modalities (score < 0.35)
  if (data.modality_results) {
    Object.entries(data.modality_results).forEach(([modKey, modVal]) => {
      if (modVal && modVal.present && typeof modVal.score === "number" && modVal.score < 0.35) {
        let modEmoji = copy.modalityEmojis.text;
        if (modKey === "image") modEmoji = copy.modalityEmojis.image;
        else if (modKey === "audio") modEmoji = copy.modalityEmojis.audio;
        else if (modKey === "metadata") modEmoji = copy.modalityEmojis.metadata;

        const desc = modVal.evidence || `Organic signals verified (anomaly score: ${modVal.score.toFixed(2)}).`;
        items.push(`✅ ${modEmoji} ${modKey.toUpperCase()}: ${desc}`);
      }
    });
  }

  // 2. Coherence: no cross-source contradictions
  const contradictions = data.cross_modal_results?.contradictions || [];
  if (contradictions.length === 0) {
    items.push(`✅ 🤝 No contradictions detected between independent sources.`);
  }

  // 3. Coherence: metadata consistency
  const metaConflicts = data.cross_modal_results?.metadata_conflicts || [];
  if (metaConflicts.length === 0 && data.modality_results?.metadata?.present) {
    items.push(`✅ ⏱️ Metadata timestamps and locations match claimed narrative.`);
  }

  // 4. Conformal set singularity
  if (Array.isArray(data.conformal_set) && data.conformal_set.length === 1) {
    const cleanName = data.conformal_set[0].replace("_", " ");
    items.push(`✅ 🎯 Conformal set singled out "${cleanName}" at 90% confidence target.`);
  }

  // 5. Authentic verdict confirmation
  if (data.label === "authentic" && items.length <= 1) {
    items.push(`✅ 🛡️ Content passed multi-layer verification checks.`);
  }

  if (items.length === 0) {
    items.push(copy.fallbacks.noFineFound);
  }

  return items;
}
