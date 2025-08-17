/* dashboard.js — read-only live dashboard pulling from /input and /state
 * Shows emotion bars, face preview (with your exact GIF assets), speaking state, pitch/rate, last chosen option.
 */
(() => {
  const API = "";
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  const bars = {
    sadness: document.getElementById("sadness-bar"),
    joy: document.getElementById("joy-bar"),
    fear: document.getElementById("fear-bar"),
    disgust: document.getElementById("disgust-bar"),
    anger: document.getElementById("anger-bar"),
    surprise: document.getElementById("surprise-bar"),
  };
  const face = document.getElementById("face");
  const speakingCell = document.getElementById("speaking-cell");
  const pitchCell = document.getElementById("pitch-cell");
  const rateCell = document.getElementById("rate-cell");
  const lastChoiceCell = document.getElementById("last-choice");

  async function postJSON(path, body) {
    const res = await fetch(API + path, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    });
    if (!res.ok) throw new Error(`${path} ${res.status}`);
    return res.json();
  }
  async function getJSON(path) {
    const res = await fetch(API + path);
    if (!res.ok) throw new Error(`${path} ${res.status}`);
    return res.json();
  }

  function setBar(el, v) { if (el) el.style.width = `${Math.max(2, Math.round((v || 0) * 100))}%`; }

  function vectorToLabel(em) {
    if (!em) return "neutral";
    const e = Object.entries(em).sort((a,b)=>b[1]-a[1]);
    const [name,val] = e[0];
    const strong = val >= 0.70;
    switch (name) {
      case "joy": return strong ? "joy_very" : "joy";
      case "sadness": return strong ? "sadness_very" : "sadness";
      case "anger": return strong ? "anger_very" : "anger";
      case "fear": return strong ? "fear_very" : "fear";
      case "surprise": return strong ? "surprise_very" : "surprise";
      case "disgust": return strong ? "disgust_very" : "disgust";
      default: return "neutral";
    }
  }
  function emotionEyesURL(label) {
    switch (label) {
      case "joy":            return './facial_expressions/animatedEyes/happyEyesSlightGIF.gif';
      case "joy_very":       return './facial_expressions/animatedEyes/happyEyesVeryGIF.gif';
      case "sadness":        return './facial_expressions/animatedEyes/sadEyesSlightGIF.gif';
      case "sadness_very":   return './facial_expressions/animatedEyes/sadEyesVeryGIF.gif';
      case "anger":          return './facial_expressions/animatedEyes/angryEyesSlightGIF.gif';
      case "anger_very":     return './facial_expressions/animatedEyes/angryEyesVeryGIF.gif';
      case "fear":           return './facial_expressions/animatedEyes/afraidEyesSlightGIF.gif';
      case "fear_very":      return './facial_expressions/animatedEyes/afraidEyesVeryGIF.gif';
      case "surprise":       return './facial_expressions/animatedEyes/surprisedEyesSlightGIF.gif';
      case "surprise_very":  return './facial_expressions/animatedEyes/surprisedEyesVeryGIF.gif';
      case "disgust":        return './facial_expressions/animatedEyes/disgustedEyesSlightGIF.gif';
      case "disgust_very":   return './facial_expressions/animatedEyes/disgustedEyesVeryGIF.gif';
      default:               return './facial_expressions/animatedEyes/neutralEyesGIF.gif';
    }
  }

  async function init() {
    try {
      // ensure backend is running (idempotent)
      await postJSON("/initialisation", { module: "frontend" });
      await postJSON("/startTheBackend", {});
    } catch {}
  }

  async function loop() {
    while (true) {
      try {
        // pull /input for live emotion + speaking state
        const payload = await postJSON("/input", {});
        const fd = payload?.frontend_data || {};
        const em = fd.emotion || {};
        setBar(bars.sadness, em.sadness);
        setBar(bars.joy, em.joy);
        setBar(bars.fear, em.fear);
        setBar(bars.disgust, em.disgust);
        setBar(bars.anger, em.anger);
        setBar(bars.surprise, em.surprise);

        // face preview uses the same eye assets as expressions page
        const label = vectorToLabel(em);
        const url = emotionEyesURL(label);
        if (face) face.style.background = `url("${url}") center / contain no-repeat`;

        speakingCell.textContent = fd.speakingState ? "Yes" : "No";
        pitchCell.textContent = typeof fd.pitch === "number" ? fd.pitch.toFixed(2) : "—";
        rateCell.textContent  = typeof fd.rate  === "number" ? fd.rate.toFixed(2)  : "—";

        // last chosen option (needs /state endpoint from earlier step)
        try {
          const st = await getJSON("/state");
          const last = st?.lastChosenOption?.label || "—";
          lastChoiceCell.textContent = last;
        } catch { /* optional */ }

      } catch (e) {
        // keep trying
      } finally {
        await sleep(300);
      }
    }
  }

  window.addEventListener("DOMContentLoaded", async () => {
    await init();
    loop();
  });
})();
