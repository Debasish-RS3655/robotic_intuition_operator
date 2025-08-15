/* frontend.js – minimal client for the conscious emotional backend
 *
 * Expected (optional) DOM:
 *  <div id="face" class="face"></div>
 *  <button id="mic-toggle"></button>
 *  <input id="text-input" placeholder="Type and press Enter" />
 *  <button id="send-btn">Send</button>
 *  <div id="log"></div>
 *  <div class="emotions">
 *    <div id="emo-sadness"></div>
 *    <div id="emo-joy"></div>
 *    <div id="emo-fear"></div>
 *    <div id="emo-disgust"></div>
 *    <div id="emo-anger"></div>
 *    <div id="emo-surprise"></div>
 *  </div>
 *
 * CSS (example idea):
 *  .face { width:160px; height:160px; background:#eee; border-radius:50%; transition:200ms; }
 *  .face.joy { background:#ffe; } .face.joy_big { transform:scale(1.05); }
 *  .face.sad { background:#eef; } .face.sad_big { filter:saturate(1.4); }
 *  ... same for anger, fear, disgust, surprise ...
 *  .face.speaking { box-shadow:0 0 0 3px rgba(0,0,0,.15) inset; }
 */

(() => {
  const API = ""; // same-origin. If you’re serving from a different host, set e.g. "http://127.0.0.1:5000"

  // ---------- helpers ----------
  const $ = (id) => document.getElementById(id);
  const faceEl = $("face");
  const micBtn = $("mic-toggle");
  const sendBtn = $("send-btn");
  const textInput = $("text-input");
  const logEl = $("log");

  const emoEls = {
    sadness: $("emo-sadness"),
    joy: $("emo-joy"),
    fear: $("emo-fear"),
    disgust: $("emo-disgust"),
    anger: $("emo-anger"),
    surprise: $("emo-surprise"),
  };

  const log = (msg) => { if (logEl) logEl.textContent = `[${new Date().toLocaleTimeString()}] ${msg}\n` + logEl.textContent; };

  async function postJSON(path, body) {
    const res = await fetch(API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    });
    if (!res.ok) throw new Error(`${path} ${res.status}`);
    return res.json();
  }

  // ---------- init handshake ----------
  async function initBackend() {
    try {
      await postJSON("/initialisation", { module: "frontend" });
      await postJSON("/startTheBackend", {});
      log("Backend initialised.");
    } catch (e) {
      log("Init error: " + e.message);
    }
  }

  // ---------- speech recognition ----------
  let rec;
  let recEnabled = false;
  let lastFinalText = "";

  function setupRecognizer() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      log("SpeechRecognition not supported in this browser.");
      return;
    }
    rec = new SR();
    rec.lang = "en-US";
    rec.continuous = true;
    rec.interimResults = true;

    rec.onresult = async (ev) => {
      // take the latest final result to avoid spamming
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const r = ev.results[i];
        const text = r[0]?.transcript?.trim();
        if (!text) continue;

        if (r.isFinal && text !== lastFinalText) {
          lastFinalText = text;
          log(`User: ${text}`);
          try {
            await postJSON("/output", { recognisedSpeech: { data: text } });
          } catch (e) {
            log("Output error: " + e.message);
          }
        }
      }
    };

    rec.onerror = (e) => {
      log("Mic error: " + e.error);
    };
    rec.onend = () => {
      if (recEnabled) {
        // auto-restart after a short pause
        setTimeout(() => { try { rec.start(); } catch(_){} }, 250);
      }
    };
  }

  function toggleMic() {
    if (!rec) return;
    recEnabled = !recEnabled;
    if (recEnabled) {
      try { rec.start(); } catch(_) {}
      if (micBtn) micBtn.textContent = "🎤 On";
      log("Mic ON");
    } else {
      try { rec.stop(); } catch(_) {}
      if (micBtn) micBtn.textContent = "🎤 Off";
      log("Mic OFF");
    }
  }

  // ---------- TTS ----------
  function speak(text, pitch, rate) {
    if (!text) return;
    if (!("speechSynthesis" in window)) {
      log("TTS not supported in this browser.");
      return;
    }
    const u = new SpeechSynthesisUtterance(text);
    if (typeof pitch === "number") u.pitch = pitch;
    if (typeof rate === "number") u.rate = rate;
    window.speechSynthesis.speak(u);
  }

  // ---------- UI updates ----------
  function setFace(token, speaking) {
    if (!faceEl) return;
    const base = "face";
    faceEl.className = `${base}${token ? " " + token : ""}${speaking ? " speaking" : ""}`;
  }

  function setEmotionsBars(em) {
    if (!em) return;
    // each bar’s width ∈ [2%, 100%] for visibility
    const width = (v) => `${Math.max(2, Math.round(v * 100))}%`;
    if (emoEls.sadness)  emoEls.sadness.style.width  = width(em.sadness || 0);
    if (emoEls.joy)      emoEls.joy.style.width      = width(em.joy || 0);
    if (emoEls.fear)     emoEls.fear.style.width     = width(em.fear || 0);
    if (emoEls.disgust)  emoEls.disgust.style.width  = width(em.disgust || 0);
    if (emoEls.anger)    emoEls.anger.style.width    = width(em.anger || 0);
    if (emoEls.surprise) emoEls.surprise.style.width = width(em.surprise || 0);
  }

  // ---------- input poll loop (backend → frontend) ----------
  let inputTimer = null;
  async function pollInput() {
    try {
      const payload = await postJSON("/input", {});
      const fd = payload?.frontend_data || {};
      const { speech, speakingState, faceExpression, pitch, rate, emotion } = fd;

      // speak if backend says so AND text present
      if (speakingState && typeof speech === "string" && speech.trim().length) {
        speak(speech, pitch, rate);
      }

      // reflect face + speaking
      setFace(faceExpression || "neutral", !!speakingState);

      // update emotion bars if present
      setEmotionsBars(emotion);

    } catch (e) {
      // keep polling even on errors
      // log("poll error: " + e.message);
    } finally {
      inputTimer = setTimeout(pollInput, 250);
    }
  }

  // ---------- typed message support ----------
  async function sendTyped() {
    if (!textInput) return;
    const text = textInput.value.trim();
    if (!text) return;
    textInput.value = "";
    log(`User: ${text}`);
    try {
      await postJSON("/output", { recognisedSpeech: { data: text } });
    } catch (e) {
      log("Output error: " + e.message);
    }
  }

  // ---------- bootstrap ----------
  window.addEventListener("DOMContentLoaded", async () => {
    setupRecognizer();
    await initBackend();
    pollInput();

    if (micBtn) {
      micBtn.textContent = "🎤 Off";
      micBtn.addEventListener("click", toggleMic);
    }
    if (sendBtn) sendBtn.addEventListener("click", sendTyped);
    if (textInput) {
      textInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); sendTyped(); }
      });
    }
  });
})();
