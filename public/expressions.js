/* expressions.js — Home: voice recognition + facial expressions + TTS
 * Uses the exact folder structure you provided.
 * Talks to your backend_conscious_full.js via /initialisation, /startTheBackend, /output, /input
 */

(() => {
  // ---------- shared ----------
  const API = "";
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const $id = (x) => document.getElementById(x);
  const logEl = $id("log");
  const log = (msg) => { if (logEl) logEl.textContent = `[${new Date().toLocaleTimeString()}] ${msg}\n` + logEl.textContent; };

  async function postJSON(path, body) {
    const res = await fetch(API + path, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    });
    if (!res.ok) throw new Error(`${path} ${res.status}`);
    return res.json();
  }

  // ---------- RIO-style globals ----------
  let startFrontend = false;              // start switch
  const testMode = false;                 // set true to disable mic loop (matches your pattern)
  const defaultPitch = 1.1;
  const defaultSpeed = 0.9;

  let responseText = "";                  // text to speak
  let current_emotion = "neutral";
  let last_emotion = "neutral";
  let rate = defaultSpeed;
  let pitch = defaultPitch;

  let voices = [];
  let ifSpoken = false;
  let blinked = false;
  const blink_time = 400;

  // asset URLs (same names as your code)
  let image_url_emotion = "./facial_expressions/animatedEyes/neutralEyesGIF.gif";
  let image_url_speech  = "./facial_expressions/animatedNoSpeeches/neutral_notSpeakGIF.gif";

  // DOM
  const emotionArea = $id("emotion_area");
  const speechArea = $id("speech_area");
  const micBtn = $id("mic-toggle");
  const startBtn = $id("start-btn");
  const sendBtn = $id("send-btn");
  const textInput = $id("text-input");
  const interim_span = $id("interim");
  const transcription = $id("speech");

  // ---------- Voice (Web Speech API) ----------
  const synth = window.speechSynthesis;

  function populateVoiceList() {
    voices = synth.getVoices() || [];
  }
  if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = populateVoiceList;
  }

  // speaking function
  function speakout(data, Ifdefault, givenPitch, givenSpeed, givenVoiceIndex=0) {
    let speakText;
    if (data === undefined) {
      speakText = new SpeechSynthesisUtterance(responseText);
      responseText = "";
    } else {
      speakText = new SpeechSynthesisUtterance(data);
    }
    if (Ifdefault !== undefined && Ifdefault !== null && Ifdefault !== true && typeof givenVoiceIndex === "number") {
      speakText.voice = voices[givenVoiceIndex];
    }
    speakText.pitch = (givenPitch !== undefined) ? givenPitch : pitch;
    speakText.rate  = (givenSpeed !== undefined) ? givenSpeed : rate;

    synth.speak(speakText);

    setTimeout(() => {
      determineSpeechImgURL();
      modifiedHTML_speech();
    }, 800);

    speakText.onpause = function (event) {
      determineNoSpeechImgURL();
      modifiedHTML_speech();
    };
    speakText.onend = function () {
      determineNoSpeechImgURL();
      modifiedHTML_speech();
    };
  }

  // recognition
  let recEnabled = false;
  let SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let speech;
  function setupRecognizer() {
    if (!SR) {
      log("SpeechRecognition not supported in this browser.");
      return;
    }
    speech = new SR();
    speech.continuous = true;
    speech.interimResults = true;
    speech.lang = "en-US";

    let lastFinal = "";
    speech.onresult = async (event) => {
      let interim_transcript = "";
      let final_transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const r = event.results[i];
        if (r.isFinal) final_transcript += r[0].transcript;
        else interim_transcript += r[0].transcript;
      }
      if (transcription) transcription.textContent = final_transcript;
      if (interim_span) interim_span.textContent = interim_transcript;

      if (final_transcript.length !== 0 && final_transcript !== lastFinal) {
        lastFinal = final_transcript;
        try {
          await postJSON("/output", { recognisedSpeech: { data: final_transcript } });
        } catch (e) {
          log("Output error: " + e.message);
        }
      }
    };
    speech.onerror = (e) => { log("Mic error: " + e.error); };
    speech.onend = () => { if (recEnabled) { try { speech.start(); } catch(_){} } };
  }
  function toggleMic() {
    if (!speech) return;
    recEnabled = !recEnabled;
    if (recEnabled) {
      try { speech.start(); } catch(_) {}
      if (micBtn) micBtn.textContent = "🎤 On";
      log("Mic ON");
    } else {
      try { speech.stop(); } catch(_) {}
      if (micBtn) micBtn.textContent = "🎤 Off";
      log("Mic OFF");
    }
  }

  // ---------- Face images (your exact mapping) ----------
  function determineEmotionURL() {
    switch (current_emotion) {
      case "joy":            image_url_emotion = './facial_expressions/animatedEyes/happyEyesSlightGIF.gif'; break;
      case "joy_very":       image_url_emotion = './facial_expressions/animatedEyes/happyEyesVeryGIF.gif'; break;
      case "sadness":        image_url_emotion = './facial_expressions/animatedEyes/sadEyesSlightGIF.gif'; break;
      case "sadness_very":   image_url_emotion = './facial_expressions/animatedEyes/sadEyesVeryGIF.gif'; break;
      case "anger":          image_url_emotion = './facial_expressions/animatedEyes/angryEyesSlightGIF.gif'; break;
      case "anger_very":     image_url_emotion = './facial_expressions/animatedEyes/angryEyesVeryGIF.gif'; break;
      case "fear":           image_url_emotion = './facial_expressions/animatedEyes/afraidEyesSlightGIF.gif'; break;
      case "fear_very":      image_url_emotion = './facial_expressions/animatedEyes/afraidEyesVeryGIF.gif'; break;
      case "surprise":       image_url_emotion = './facial_expressions/animatedEyes/surprisedEyesSlightGIF.gif'; break;
      case "surprise_very":  image_url_emotion = './facial_expressions/animatedEyes/surprisedEyesVeryGIF.gif'; break;
      case "disgust":        image_url_emotion = './facial_expressions/animatedEyes/disgustedEyesSlightGIF.gif'; break;
      case "disgust_very":   image_url_emotion = './facial_expressions/animatedEyes/disgustedEyesVeryGIF.gif'; break;
      default:               image_url_emotion = './facial_expressions/animatedEyes/neutralEyesGIF.gif'; break;
    }
  }
  function determineSpeechImgURL() {
    switch (current_emotion) {
      case "joy":
      case "joy_very":       image_url_speech = './facial_expressions/animatedSpeeches/happySpeakingRAW.gif'; break;
      case "sadness":
      case "sadness_very":   image_url_speech = './facial_expressions/animatedSpeeches/sadSpeakingRAW.gif'; break;
      case "fear":
      case "fear_very":
      case "anger":
      case "anger_very":     image_url_speech = './facial_expressions/animatedSpeeches/afraid_angrySpeakingRAW.gif'; break;
      case "surprise":
      case "surprise_very":
      case "disgust":
      case "disgust_very":   image_url_speech = './facial_expressions/animatedSpeeches/surprised_disgustSpeakingRAW.gif'; break;
      default:               image_url_speech = './facial_expressions/animatedSpeeches/neutralSpeakingRAW.gif'; break;
    }
  }
  function determineNoSpeechImgURL() {
    switch (current_emotion) {
      case "joy":
      case "joy_very":       image_url_speech = './facial_expressions/animatedNoSpeeches/happy_notSpeakGIF.gif'; break;
      case "sadness":
      case "sadness_very":   image_url_speech = './facial_expressions/animatedNoSpeeches/sad_notSpeakGIF.gif'; break;
      case "fear":
      case "fear_very":
      case "anger":
      case "anger_very":     image_url_speech = './facial_expressions/animatedNoSpeeches/angry_afraid_notSpeakGIF.gif'; break;
      case "surprise":
      case "surprise_very":
      case "disgust":
      case "disgust_very":   image_url_speech = './facial_expressions/animatedNoSpeeches/surprised_disgust_notSpeakGIF.gif'; break;
      default:               image_url_speech = './facial_expressions/animatedNoSpeeches/neutral_notSpeakGIF.gif'; break;
    }
  }
  function modifiedHTML_emotion() {
    if (emotionArea) emotionArea.style.background = `url("${image_url_emotion}") no-repeat center / contain`;
  }
  function modifiedHTML_speech() {
    if (speechArea) speechArea.style.background = `url("${image_url_speech}") no-repeat center / contain`;
  }
  function changeEmotionState() {
    if (current_emotion !== last_emotion) {
      if (!blinked) {
        image_url_emotion = "./facial_expressions/animatedEyes/blinkFinalPNG.png";
        modifiedHTML_emotion();
        blinked = true;
        determineEmotionURL();
        setTimeout(() => { modifiedHTML_emotion(); blinked = false; }, blink_time);
      }
      last_emotion = current_emotion;
    }
  }

  // map backend emotion vector → label family used above
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

  // ---------- backend IO loop ----------
  async function pollInput() {
    while (true) {
      try {
        const payload = await postJSON("/input", {});
        const fd = payload?.frontend_data || {};
        const { speech, speakingState, pitch: p, rate: r, emotion } = fd;

        // update emotion → GIFs (eyes)
        const label = vectorToLabel(emotion);
        current_emotion = label;
        changeEmotionState();

        // if backend queued speech & told us to speak, do it with pitch/rate
        if (speakingState && typeof speech === "string" && speech.trim().length) {
          if (typeof p === "number") pitch = p;
          if (typeof r === "number") rate  = r;
          responseText = speech;
          speakout();
        }

      } catch (e) {
        // keep looping even on error
      } finally {
        await sleep(250);
      }
    }
  }

  // ---------- typed send ----------
  async function sendTyped() {
    const el = $id("text-input");
    if (!el) return;
    const t = el.value.trim();
    if (!t) return;
    el.value = "";
    try {
      await postJSON("/output", { recognisedSpeech: { data: t } });
    } catch (e) {
      log("Output error: " + e.message);
    }
  }

  // ---------- bootstrap ----------
  window.addEventListener("DOMContentLoaded", async () => {
    populateVoiceList();
    setupRecognizer();

    try {
      await postJSON("/initialisation", { module: "frontend" });
      await postJSON("/startTheBackend", {});
      log("Backend initialised.");
    } catch (e) {
      log("Init error: " + e.message);
    }

    // default faces on load
    determineEmotionURL();  modifiedHTML_emotion();
    determineNoSpeechImgURL(); modifiedHTML_speech();

    // UI
    if (micBtn) micBtn.addEventListener("click", toggleMic);
    if (startBtn) startBtn.addEventListener("click", () => { startFrontend = true; log("Started."); });
    if (sendBtn) sendBtn.addEventListener("click", sendTyped);
    if (textInput) textInput.addEventListener("keydown", (e)=>{ if (e.key==="Enter"){ e.preventDefault(); sendTyped(); } });

    // start loops
    startFrontend = true;
    pollInput();
  });
})();
