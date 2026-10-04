/**
 * TrustLayer Extension - Visible UI Copy
 * Central repository for all user-facing strings and copy used in the extension popup.
 */

const POPUP_COPY = {
  header: {
    title: "TrustLayer",
    subtitle: "Multimodal Digital Content Verification"
  },

  states: {
    empty: "🔎 Select some text on a page and right-click to check it.",
    analyzing: "⏳ Analyzing selected text with TrustLayer...",
    errorUnreachable: "📡 TrustLayer can't reach the server. Make sure the backend is running.",
    errorGeneric: "📡 TrustLayer encountered an error during analysis."
  },

  verdicts: {
    authentic: {
      emoji: "✅",
      headline: "Authentic"
    },
    manipulated: {
      emoji: "🎭",
      headline: "Manipulated"
    },
    coordinated_synthetic: {
      emoji: "🧩",
      headline: "Coordinated Synthetic"
    },
    insufficient_evidence: {
      emoji: "🤷",
      headline: "Insufficient Evidence"
    }
  },

  sections: {
    howWeGotHere: "How we got here",
    risky: "⚠️ What's risky to trust",
    fine: "✅ What looks fine",
    checkedSnippet: "Checked text"
  },

  meta: {
    confidence: "Calibrated confidence",
    abstainedBadge: "⚠️ Abstained",
    conformalSet: "Prediction set"
  },

  button: {
    openFullReport: "📋 Open full report"
  },

  modalityEmojis: {
    text: "📝",
    image: "🖼️",
    audio: "🎙️",
    metadata: "🏷️",
    cross_modal: "🔗",
    contradiction: "⚡",
    guidance: "🧭",
    fine: "✅",
    risky: "⚠️",
    neutral: "🔍"
  },

  fallbacks: {
    noRisksFound: "🛡️ No manipulation markers or contradictions detected.",
    noFineFound: "🔍 No positive corroborating markers found.",
    noFindingsAvailable: "🤷 No detailed breakdown available for this bundle."
  }
};

if (typeof window !== "undefined") {
  window.POPUP_COPY = POPUP_COPY;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = POPUP_COPY;
}
