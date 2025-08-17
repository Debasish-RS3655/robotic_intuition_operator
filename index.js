// backend_conscious_full.js
// Fully conscious + emotional chatbot backend with:
// - complete emotional decision functions (inclination, repulsion, guilt/prohibited, multi-option choice)
// - NLU→stimulus, speak/abstain, option selection, anticipated & guilt stimuli
// - long-term memory store (persisted JSON)
// - frontend compatibility (/initialisation, /startTheBackend, /output, /input, /restore, /save)
// - emotion_sphere.js is used unchanged

"use strict";

/* =============================
 * Dependencies & Core Engine
 * ============================= */
const express = require("express");
const bodyParser = require("body-parser");
const crypto = require("crypto");
const fs = require("fs");
const path = require("path");


// natural (tokenization + stemming)
const natural = require('natural');
const tokenizer = new natural.TreebankWordTokenizer();   // good, conservative English tokenizer
const stemmer = natural.PorterStemmer;                  // Porter stemmer (English)


// --- Word2Vec embeddings (semantic similarity) ---
const word2vec = require('word2vec');
let W2V = null;                     // model instance once loaded
let EMOTION_CENTROIDS = null;       // { joy: Float32Array, sadness: ... }

// load model once; keep server usable with graceful fallback
word2vec.loadModel(path.join(__dirname, 'word2vec', 'text8.bin'), (err, model) => {
  if (err) { console.warn('Word2Vec model not loaded, falling back to lexical:', err.message); return; }
  W2V = model;
  buildEmotionCentroids();          // compute emotion centroids once model is ready
  console.log('✅ Word2Vec model loaded; semantic emotion scoring enabled.');
});



// Optional (only if you set WATSON_NLU_APIKEY & WATSON_NLU_URL)
let NaturalLanguageUnderstandingV1 = null;
let IamAuthenticator = null;
try {
  ({ default: NaturalLanguageUnderstandingV1 } = require("ibm-watson/natural-language-understanding/v1"));
  ({ IamAuthenticator } = require("ibm-watson/auth"));
} catch (_) { /* optional */ }

// Core emotional engine (leave this file untouched)
const emotion = require("./emotion_sphere.js");

/* =============================
 * Config
 * ============================= */
const PORT = process.env.PORT || 5000;
const MEMORY_FILE = path.join(__dirname, "memory_longterm.json");
const testMode = !!process.env.TEST_MODE; // true to print more logs if you want

/* =============================
 * Agent Profile & State
 * ============================= */
const Agent = {
  name: "Elsa",
  bigFive: {
    openness: 0.5, conscientiousness: 0.5, extraversion: 0.5, agreeableness: 0.5, neuroticism: 0.5
  },
  started: false
};

// People known to the agent (speaker registry)
const personsPresent = []; // [{name, likeness, personality}]
const defaultLikeness = 0.5;

// Frontend envelope (shape kept for compatibility)
const F = {
  audio_incoming: { text: null, val: null, json: null },
  song_incoming: { text: null, val: null, json: null },           // kept for shape
  speaker_incoming: { text: null, val: null, json: null },        // kept for shape
  visual_incoming: { text: null, val: null, json: null },
  verbal_incoming: { text: null, val: null, json: [] },
  pose_incoming: { text: null, val: null, json: [] },
  frontend_data: {
    speech: null,
    emotion: null,
    emotionScore: null,
    rate: null,
    pitch: null,
    speakingState: null,
    faceExpression: null
  },
  song_incomingJS: { text: null, val: null, json: null }
};

let queueSpeech = "";

/* =============================
 * Utilities
 * ============================= */
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const map = (x, inMin, inMax, outMin, outMax) =>
  outMin + (clamp01((x - inMin) / (inMax - inMin)) * (outMax - outMin));

// ---------- Word2Vec helpers ----------
function w2vVec(word) {
  if (!W2V) return null;
  try { return W2V.getVector(String(word).toLowerCase()).values; } catch { return null; }
}
function addInPlace(a, b) { for (let i=0;i<a.length;i++) a[i]+=b[i]; return a; }
function scaleInPlace(a, s) { for (let i=0;i<a.length;i++) a[i]*=s; return a; }
function avgVec(words) {
  if (!W2V) return null;
  let acc = null, n = 0;
  for (const w of words) {
    const v = w2vVec(w);
    if (!v) continue;
    if (!acc) acc = new Float32Array(v); else addInPlace(acc, v);
    n++;
  }
  if (!acc || n === 0) return null;
  return scaleInPlace(acc, 1/n);
}
function cosine(a, b) {
  if (!a || !b) return 0;
  let dot=0, na=0, nb=0;
  for (let i=0;i<a.length;i++){ const x=a[i], y=b[i]; dot+=x*y; na+=x*x; nb+=y*y; }
  if (na===0 || nb===0) return 0;
  return dot / (Math.sqrt(na)*Math.sqrt(nb));
}

// emotion seeds: reuse your LEX lists (already defined below)
function buildEmotionCentroids() {
  if (!W2V) return;
  EMOTION_CENTROIDS = {};
  for (const k of Object.keys(LEX)) {
    const centroid = avgVec(LEX[k]);
    if (centroid) EMOTION_CENTROIDS[k] = centroid;
  }
}

// Turn a text into an averaged sentence vector
function sentenceVector(text) {
  if (!W2V) return null;
  const toks = tokenizer.tokenize(text || '').map(normalizeToken).filter(isMeaningfulToken);
  return avgVec(toks.length ? toks : [String(text || '')]);
}


// Context-vs-token weighting for emotions
const TOKEN_WEIGHT = 0.2;
const CONTEXT_WEIGHT = 0.8;

function blendEmotions(a, b, wa = TOKEN_WEIGHT, wb = CONTEXT_WEIGHT) {
  const ax = a || {}, bx = b || {};
  const clamp = (v) => Math.max(0, Math.min(1, v || 0));
  return {
    sadness: clamp((ax.sadness || 0) * wa + (bx.sadness || 0) * wb),
    joy: clamp((ax.joy || 0) * wa + (bx.joy || 0) * wb),
    fear: clamp((ax.fear || 0) * wa + (bx.fear || 0) * wb),
    disgust: clamp((ax.disgust || 0) * wa + (bx.disgust || 0) * wb),
    anger: clamp((ax.anger || 0) * wa + (bx.anger || 0) * wb),
    surprise: clamp((ax.surprise || 0) * wa + (bx.surprise || 0) * wb),
  };
}





const hashLabel = (prefix, text) =>
  `${prefix}_${crypto.createHash("sha1").update(String(text)).digest("hex").slice(0, 8)}`;

function bubblSrt(arr, key) { // simple stable-ish ascending bubble sort for small arrays
  const a = arr.slice();
  for (let i = 0; i < a.length - 1; i++) {
    for (let j = 0; j < a.length - i - 1; j++) {
      if (a[j][key] > a[j + 1][key]) {
        const t = a[j]; a[j] = a[j + 1]; a[j + 1] = t;
      }
    }
  }
  return a;
}

function getPersonIndex(name) {
  for (let i = 0; i < personsPresent.length; i++) if (personsPresent[i].name === name) return i;
  return -1;
}
function ensureSpeaker(name) {
  let idx = getPersonIndex(name);
  if (idx === -1) {
    personsPresent.push({ name, likeness: defaultLikeness, personality: {} });
    idx = personsPresent.length - 1;
  }
  return idx;
}

function isPositiveSurprise(optionObj) {
  // Heuristic: surprise considered positive if joy is dominant over average negatives
  const negAvg = (optionObj.sadness + optionObj.fear + optionObj.disgust + optionObj.anger) / 4;
  return (optionObj.joy >= negAvg);
}

function checkIfBasicEmotionStim(label) {
  const basics = new Set([
    "sadness__", "joy__", "fear__", "disgust__", "anger__", "surprise__",
    "sadness_", "joy_", "fear_", "disgust_", "anger_", "surprise_"
  ]);
  return basics.has(label);
}

function checkIfSpecialDecisions(givenOption) {
  switch (givenOption) {
    case "wait_": case "speak_": case "chooseOption_": case "trackFace_":
    case "ignoreFace_": case "sayWhatSeen_": case "sayPersonality_":
    case "singSong_": case "emotion_choose_":
    case "sadness_": case "joy_": case "fear_": case "disgust_": case "anger_": case "surprise_":
      return true;
    default: return false;
  }
}

/* =============================
 * Long-term Memory (persisted)
 * ============================= */
function removeFromMemory(label) {
  const mem = readMemory();
  if (!Array.isArray(mem.longTerm)) mem.longTerm = [];
  const idx = mem.longTerm.findIndex(x => x && x.label === label);
  if (idx === -1) return false;
  mem.longTerm.splice(idx, 1);
  writeMemory(mem);
  return true;
}

function readMemory() {
  if (!fs.existsSync(MEMORY_FILE)) {
    fs.writeFileSync(MEMORY_FILE, JSON.stringify({ longTerm: [] }, null, 2));
  }
  return JSON.parse(fs.readFileSync(MEMORY_FILE, "utf8"));
}
function writeMemory(mem) {
  fs.writeFileSync(MEMORY_FILE, JSON.stringify(mem, null, 2));
}

// memoryRet(type:'longTerm', label, set?, data?, createIfMissing?, ensureExist?, relatedStimuli?, relatedPeople?, relatedEmotions?, index?)
function memoryRet(type, label, setFlag = false, data = null, createIfMissing = false, ensureExist = false, relatedStimuli = null, relatedPeople = null, relatedEmotions = null, indexOpt = null) {
  const mem = readMemory();
  if (!mem[type]) mem[type] = [];
  const arr = mem[type];

  let idx = arr.findIndex(x => x && x.label === label);
  if (idx === -1 && (createIfMissing || ensureExist)) {
    const newObj = {
      label,
      anger: 0, joy: 0, sadness: 0, fear: 0, disgust: 0, surprise: 0,
      timesOccurred: 0, totalTimesUsed: 0, isProhibited: false,
      openness: Agent.bigFive.openness,
      conscientiousness: Agent.bigFive.conscientiousness,
      extraversion: Agent.bigFive.extraversion,
      agreeableness: Agent.bigFive.agreeableness,
      neuroticism: Agent.bigFive.neuroticism,
      relatedTasks: {},
      relatedStimuli: {},
      relatedPeople: {},
      time: Date.now(),
      indexUnderLabel: 0
    };
    if (relatedStimuli) for (const k of relatedStimuli) newObj.relatedStimuli[k] = true;
    if (relatedPeople) for (const k of relatedPeople) newObj.relatedPeople[k] = true;
    if (relatedEmotions && Array.isArray(relatedEmotions) && relatedEmotions.length === 6) {
      [newObj.sadness, newObj.joy, newObj.fear, newObj.disgust, newObj.anger, newObj.surprise] = relatedEmotions.map(clamp01);
    }
    arr.push(newObj);
    writeMemory(mem);
    idx = arr.length - 1;
  }
  if (setFlag && data) {
    // replace or insert at given index if provided
    if (indexOpt != null && arr[indexOpt] && arr[indexOpt].label === label) {
      arr[indexOpt] = Object.assign({}, data);
    } else {
      if (idx === -1) arr.push(Object.assign({}, data)); else arr[idx] = Object.assign({}, data);
    }
    writeMemory(mem);
  }
  if (idx === -1) return undefined;
  return { index: idx, data: arr[idx] };
}

/* =============================
 * NLU Adapter (Watson or fallback)
 * ============================= */
let watsonNLU = null;
function getNLU() {
  if (watsonNLU !== null) return watsonNLU;
  const apikey = process.env.WATSON_NLU_APIKEY;
  const url = process.env.WATSON_NLU_URL;
  if (!apikey || !url || !NaturalLanguageUnderstandingV1) {
    watsonNLU = false; // no NLU, fallback
    return watsonNLU;
  }
  watsonNLU = new NaturalLanguageUnderstandingV1({
    version: "2023-10-10",
    authenticator: new IamAuthenticator({ apikey }),
    serviceUrl: url
  });
  return watsonNLU;
}

const LEX = {
  joy: ["happy", "glad", "great", "awesome", "love", "amazing", "excellent", "win", "yay", "good"],
  sadness: ["sad", "down", "unhappy", "depressed", "sorrow", "cry", "gloom", "bad", "loss", "lonely"],
  anger: ["angry", "mad", "furious", "annoyed", "rage", "irritated", "hate", "disrespect"],
  fear: ["scared", "afraid", "anxious", "worried", "panic", "nervous", "concerned", "fear"],
  disgust: ["disgust", "gross", "nasty", "eww", "filthy", "yuck", "repulsive", "vomit"],
  surprise: ["surprised", "shocked", "wow", "unexpected", "sudden", "unbelievable"]
};

function kwScores(textOrToken) {
  // If W2V not ready, fall back to the old lexical heuristic
  if (!W2V || !EMOTION_CENTROIDS) {
    // ---- legacy fallback (your old code) ----
    const t = (textOrToken || "").toLowerCase();
    const s = { sadness: 0, joy: 0, fear: 0, disgust: 0, anger: 0, surprise: 0 };
    for (const w of t.split(/\W+/)) {
      if (!w) continue;
      if (LEX.joy.includes(w)) s.joy += 0.2;
      if (LEX.sadness.includes(w)) s.sadness += 0.2;
      if (LEX.anger.includes(w)) s.anger += 0.2;
      if (LEX.fear.includes(w)) s.fear += 0.2;
      if (LEX.disgust.includes(w)) s.disgust += 0.2;
      if (LEX.surprise.includes(w)) s.surprise += 0.2;
    }
    for (const k in s) s[k] = clamp01(s[k]);
    if (Object.values(s).every(v => v === 0)) s.joy = 0.05;
    return s;
  }

  // ---- semantic scoring with word2vec ----
  const v = w2vVec(textOrToken) || sentenceVector(textOrToken);
  const out = { sadness:0, joy:0, fear:0, disgust:0, anger:0, surprise:0 };
  if (!v) { out.joy = 0.05; return out; }

  // cosine similarity to each emotion centroid; clamp negatives to 0
  for (const k of Object.keys(out)) {
    const c = EMOTION_CENTROIDS[k];
    if (!c) { out[k] = 0; continue; }
    const sim = cosine(v, c);          // [-1, 1]
    out[k] = clamp01(Math.max(0, sim)); // keep only positive similarity
  }

  // normalize so strongest dimension is ≤ 1; tiny nudge if all zero
  const mx = Math.max(...Object.values(out), 0);
  if (mx === 0) out.joy = 0.05;
  else for (const k of Object.keys(out)) out[k] = out[k] / mx; // scale to [0..1] peak=1
  return out;
}


// -------- token helpers via natural --------
// use natural's stopwords if present; add a few UI/common extras
const NATURAL_STOPWORDS = (natural.stopwords || []).map(s => s.toLowerCase());
const EXTRA_STOPWORDS = [
  "ok", "okay", "yeah", "yep", "nope", "uh", "um", "hi", "hello", "hey", "please", "thanks", "thank", "you"
];
const STOPWORDS = new Set([...NATURAL_STOPWORDS, ...EXTRA_STOPWORDS]);

function normalizeToken(t) {
  return String(t).toLowerCase().replace(/[^a-z0-9_+/-]/g, "").trim();
}

function isMeaningfulToken(t) {
  if (!t || t.length < 2) return false;
  if (STOPWORDS.has(t)) return false;
  // avoid colliding with special engine labels
  if (checkIfSpecialDecisions(t)) return false;
  return true;
}

/**
 * Build stimuli from tokens (canonicalized by stemming).
 * - Label = stem (canonical form) so "running/run/ran" map to one memory key.
 * - Emotion = kwScores() on token and stem (max of both).
 */
async function tokenStimuliFromText(text, personality) {
  const rawTokens = tokenizer.tokenize(text || "");
  const sentenceEmo = await sentenceEmotionFromText(text); // ← whole-sentence context once
  const seen = new Set();
  const stimuli = [];

  for (const raw of rawTokens) {
    const tok = normalizeToken(raw);
    if (!isMeaningfulToken(tok)) continue;

    const stem = stemmer.stem(tok);
    const label = normalizeToken(stem);
    if (!label || seen.has(label)) continue;
    seen.add(label);

    // context-aware emotions for both surface and stem
    const emoTok = kwScoresContext(tok, sentenceEmo);
    const emoStem = kwScoresContext(label, sentenceEmo);

    // pick the stronger signal dimension-wise
    const emo = {
      sadness: Math.max(emoTok.sadness, emoStem.sadness),
      joy: Math.max(emoTok.joy, emoStem.joy),
      fear: Math.max(emoTok.fear, emoStem.fear),
      disgust: Math.max(emoTok.disgust, emoStem.disgust),
      anger: Math.max(emoTok.anger, emoStem.anger),
      surprise: Math.max(emoTok.surprise, emoStem.surprise),
    };

    // tiny neutral nudge if totally flat
    if (Object.values(emo).every(v => (v || 0) === 0)) emo.joy = 0.05;

    const stim = makeStimulus(label, emo, personality || {}, { isProhibited: false });
    stim.relatedStimuli = Object.assign({}, stim.relatedStimuli, { [tok]: true });
    stimuli.push(stim);
  }
  return stimuli;
}



async function analyzeNLU(text) {
  const nlu = getNLU();
  if (nlu === false) {
    return { emotion: kwScores(text), intents: [], entities: [] };
  }
  try {
    const res = await nlu.analyze({
      text,
      features: {
        sentiment: {},
        emotion: {},
        keywords: { emotion: true, sentiment: true, limit: 8 },
        concepts: { limit: 5 },
        categories: {}
      }
    });
    const e = res.result?.emotion?.document?.emotion || {};
    const emotionOut = {
      sadness: clamp01(e.sadness ?? 0),
      joy: clamp01(e.joy ?? 0),
      fear: clamp01(e.fear ?? 0),
      disgust: clamp01(e.disgust ?? 0),
      anger: clamp01(e.anger ?? 0),
      surprise: kwScores(text).surprise // Watson has no surprise; blend fallback
    };
    const entities = (res.result.keywords || []).map(k => k.text.toLowerCase());
    return { emotion: emotionOut, intents: [], entities };
  } catch (err) {
    console.warn("NLU error, using fallback:", err.message);
    return { emotion: kwScores(text), intents: [], entities: [] };
  }
}

async function sentenceEmotionFromText(fullText) {
  const nlu = getNLU();
  if (nlu !== false && nlu) {
    try {
      const res = await analyzeNLU(fullText);
      return res?.emotion || { sadness:0, joy:0, fear:0, disgust:0, anger:0, surprise:0 };
    } catch { /* fall through to w2v */ }
  }
  // Word2Vec sentence emotion (fallback or primary when NLU absent)
  if (W2V && EMOTION_CENTROIDS) {
    const sv = sentenceVector(fullText);
    if (sv) {
      const out = {};
      for (const k of Object.keys(LEX)) {
        const c = EMOTION_CENTROIDS[k];
        out[k] = c ? clamp01(Math.max(0, cosine(sv, c))) : 0;
      }
      const mx = Math.max(...Object.values(out), 0);
      return mx === 0 ? { sadness:0, joy:0.05, fear:0, disgust:0, anger:0, surprise:0 }
                      : Object.fromEntries(Object.entries(out).map(([k,v]) => [k, v/mx]));
    }
  }
  return { sadness:0, joy:0.05, fear:0, disgust:0, anger:0, surprise:0 };
}

function kwScoresContext(labelOrStem, sentenceEmo) {
  // Token/local score via semantic kwScores (w2v if loaded, else lexical)
  const local = kwScores(labelOrStem);

  // If you have Watson emotion (from analyzeNLU), blend as-is.
  // Otherwise, derive a sentence-level emotion via Word2Vec sentence vector.
  let context = sentenceEmo;
  if ((!context || typeof context !== 'object') && W2V) {
    const sv = sentenceVector(labelOrStem); // NOTE: if you want whole utterance, pass the whole sentence instead where you call this
    if (sv && EMOTION_CENTROIDS) {
      const temp = {};
      for (const k of Object.keys(local)) {
        const c = EMOTION_CENTROIDS[k]; temp[k] = c ? clamp01(Math.max(0, cosine(sv, c))) : 0;
      }
      // normalize peak to 1
      const mx = Math.max(...Object.values(temp), 0);
      context = mx === 0 ? { sadness:0, joy:0.05, fear:0, disgust:0, anger:0, surprise:0 }
                         : Object.fromEntries(Object.entries(temp).map(([k,v]) => [k, v/mx]));
    }
  }
  return blendEmotions(local, context || { sadness:0, joy:0, fear:0, disgust:0, anger:0, surprise:0 }, TOKEN_WEIGHT, CONTEXT_WEIGHT);
}




/* =============================
 * Face / Voice hints
 * ============================= */
function faceFromEmotions(e) {
  const entries = Object.entries(e).sort((a, b) => b[1] - a[1]);
  const [maxName, maxVal] = entries[0];
  let face = "neutral";
  switch (maxName) {
    case "joy": face = maxVal > 0.7 ? "joy_big" : "joy"; break;
    case "sadness": face = maxVal > 0.7 ? "sad_big" : "sad"; break;
    case "anger": face = maxVal > 0.7 ? "anger_big" : "anger"; break;
    case "fear": face = maxVal > 0.7 ? "fear_big" : "fear"; break;
    case "disgust": face = maxVal > 0.7 ? "disgust_big" : "disgust"; break;
    case "surprise": face = maxVal > 0.7 ? "surprise_big" : "surprise"; break;
    default: face = "neutral";
  }
  const pitch = Math.max(0.6, Math.min(1.4, 0.9 + 0.2 * (e.joy + e.surprise) - 0.1 * (e.sadness + e.anger + e.fear)));
  const rate = Math.max(0.7, Math.min(1.3, 0.9 + 0.2 * (e.surprise + e.joy) - 0.15 * (e.sadness)));
  return { face, pitch, rate };
}

/* =============================
 * Decision Speech Extractor (lightweight)
 * ============================= */
const decisionSpeechExtractor = {
  decisionSpeech(text) {
    if (!text) return null;
    const t = text.toLowerCase();

    // patterns like: "should I X or Y?", "do I choose A or B", "I'll go for A or B", "prefer A or B"
    const orMatch = t.match(/(?:should i|do i|shall i|would i|will i|i (?:will|should|shall|would)?\s*(?:go for|choose|pick|select|prefer))\s+([^?]+?)(?:\s+or\s+)([^?]+?)(?:[.?!]|$)/i);
    if (orMatch) {
      const left = orMatch[1].trim().replace(/^to\s+/, "");
      const right = orMatch[2].trim().replace(/^to\s+/, "");
      return { verbs: ["chooseOption_"], objects: [left, right] };
    }

    // "should I <verb>" (single option implied)
    const singleVerb = t.match(/should i\s+(?:to\s+)?([a-z][a-z ]+?)(?:[.?!]|$)/i);
    if (singleVerb) {
      return { verbs: [singleVerb[1].trim().split(/\s+/)[0]], objects: [] };
    }

    // "choose/go for/pick/select <object>"
    const simpleChoose = t.match(/(?:choose|go for|pick|select|prefer)\s+([^.?]+)(?:[.?!]|$)/i);
    if (simpleChoose) {
      const obj = simpleChoose[1].trim();
      return { verbs: ["chooseOption_"], objects: [obj] };
    }

    return null;
  }
};

/* =============================
 * Emotional Decision Constants
 * (mirrors your "latest scientific version")
 * ============================= */
const inclinationWeight = 0.40;
const repulsionWeight = 0.40;
const personalitySimilarityWeight = 0.20;
const currentRepulsionWeight = 0.60;
const currentAntiInclinationWeight = 0.60;
const extremeEmotionThreshold = 0.80;
const abstentionConstant = 0.45;
const guiltInfluenceWeight = 0.35;
const choiceThreshold = 0.30;

// dynamic weight vectors (ascending importance) depending on # of dependent emotions
function weightsFor(n) {
  // normalized ascending 1..n
  const arr = Array.from({ length: n }, (_, i) => i + 1);
  const s = arr.reduce((a, b) => a + b, 0);
  return arr.map(v => v / s);
}

/* =============================
 * System emotion mirrors (rio_*)
 * ============================= */
let rio_sadness = 0, rio_joy = 0, rio_fear = 0, rio_disgust = 0, rio_anger = 0, rio_surprise = 0;
function refreshSystemEmotions() {
  const e = emotion.getEmotions();
  rio_sadness = e.sadness; rio_joy = e.joy; rio_fear = e.fear; rio_disgust = e.disgust; rio_anger = e.anger; rio_surprise = e.surprise;
}

/* =============================
 * Personality similarity helpers
 * ============================= */
function PersonalityDiffScaledScore(x, isObj = false) {
  // When x is a person object: x.personality has Big-Five
  const p = isObj ? x.personality : (x && x.personality ? x.personality : x);
  const keys = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"];
  let dot = 0, na = 0, nb = 0;
  for (const k of keys) {
    const a = clamp01(Agent.bigFive[k] ?? 0.5);
    const b = clamp01(p?.[k] ?? 0.5);
    dot += a * b; na += a * a; nb += b * b;
  }
  if (na === 0 || nb === 0) return 0.5;
  return clamp01(dot / (Math.sqrt(na) * Math.sqrt(nb)));
}
function personalityNearby(optionObj, returnScore) {
  // compare stimulus' embedded personality with agent's
  const p = {
    openness: optionObj.openness,
    conscientiousness: optionObj.conscientiousness,
    extraversion: optionObj.extraversion,
    agreeableness: optionObj.agreeableness,
    neuroticism: optionObj.neuroticism
  };
  return PersonalityDiffScaledScore({ personality: p }, true);
}

/* =============================
 * Full Repulsion & Inclination
 * ============================= */
let extraDisplayControl = []; // used to avoid duplicate logs in test mode

function calculateRepulsion(optionObj, reverseScale = false) {
  refreshSystemEmotions();
  let sadnessInvolved = optionObj.sadness !== "system_sadness_";
  let joyInvolved = optionObj.joy !== "system_joy_";
  let fearInvolved = optionObj.fear !== "system_fear_";
  let disgustInvolved = optionObj.disgust !== "system_disgust_";
  let angerInvolved = optionObj.anger !== "system_anger_";
  let surpriseInvolved = optionObj.surprise !== "system_surprise_";

  let sadnessDiff, joyDiff, fearDiff, disgustDiff, angerDiff, surpriseDiff;

  if (sadnessInvolved) sadnessDiff = optionObj.sadness > rio_sadness ? optionObj.sadness - rio_sadness : 0;
  if (joyInvolved) joyDiff = rio_joy > optionObj.joy ? rio_joy - optionObj.joy : 0;
  if (fearInvolved) fearDiff = optionObj.fear > rio_fear ? optionObj.fear - rio_fear : 0;
  if (disgustInvolved) disgustDiff = optionObj.disgust > rio_disgust ? optionObj.disgust - rio_disgust : 0;
  if (angerInvolved) angerDiff = optionObj.anger > rio_anger ? optionObj.anger - rio_anger : 0;
  if (surpriseInvolved) {
    if (isPositiveSurprise(optionObj)) {
      surpriseDiff = rio_surprise > optionObj.surprise ? rio_surprise - optionObj.surprise : 0;
    } else {
      surpriseDiff = optionObj.surprise > rio_surprise ? optionObj.surprise - rio_surprise : 0;
    }
  }

  const diffArr = [sadnessDiff, joyDiff, fearDiff, disgustDiff, angerDiff, surpriseDiff];
  const labels = ['sadness', 'joy', 'fear', 'disgust', 'anger', 'surprise'];

  for (let i = 0; i < diffArr.length; i++) {
    if (diffArr[i] === undefined) {
      if (testMode) console.log(optionObj.label, "'s repulsion is independent of ", labels[i]);
      diffArr.splice(i, 1); labels.splice(i, 1); i--;
    }
  }

  let weightedArray = weightsFor(diffArr.length);
  // ascending sort by value with labels paired
  for (let i = 0; i < diffArr.length - 1; i++) {
    for (let j = i + 1; j < diffArr.length; j++) {
      if (diffArr[i] > diffArr[j]) {
        const t = diffArr[i]; diffArr[i] = diffArr[j]; diffArr[j] = t;
        const t2 = labels[i]; labels[i] = labels[j]; labels[j] = t2;
      }
    }
  }

  let w = { sadness: 0, joy: 0, fear: 0, disgust: 0, anger: 0, surprise: 0 };
  for (let i = 0; i < diffArr.length; i++) {
    w[labels[i]] = diffArr[i] * weightedArray[i];
  }

  let weightedAvg = w.sadness + w.joy + w.fear + w.disgust + w.anger + w.surprise;

  if (optionObj.hasOwnProperty("guilt_influence")) {
    if (testMode) console.log("Guilt influence of ", optionObj.label, ": ", optionObj.guilt_influence);
    weightedAvg = weightedAvg * currentRepulsionWeight + optionObj.guilt_influence * guiltInfluenceWeight;
  }

  if (reverseScale) {
    const rev = map(weightedAvg, 0, 1, 1, 0);
    return rev;
  }
  return weightedAvg;
}

function hasSufficientEmWeight(givenOption, returnTheScore = false) {
  refreshSystemEmotions();

  function calculateInclination(optionObj, sadIncl, joyIncl, fearIncl, disgustIncl, angerIncl, surpriseIncl) {
    let sadnessInvolved = optionObj.sadness !== "system_sadness_";
    let joyInvolved = optionObj.joy !== "system_joy_";
    let fearInvolved = optionObj.fear !== "system_fear_";
    let disgustInvolved = optionObj.disgust !== "system_disgust_";
    let angerInvolved = optionObj.anger !== "system_anger_";
    let surpriseInvolved = optionObj.surprise !== "system_surprise_";

    let sadnessDiff, joyDiff, fearDiff, disgustDiff, angerDiff, surpriseDiff;

    if (sadnessInvolved) sadnessDiff = rio_sadness > optionObj.sadness ? rio_sadness - optionObj.sadness : 0;
    if (joyInvolved) joyDiff = optionObj.joy > rio_joy ? optionObj.joy - rio_joy : 0;
    if (fearInvolved) fearDiff = rio_fear > optionObj.fear ? rio_fear - optionObj.fear : 0;
    if (disgustInvolved) disgustDiff = rio_disgust > optionObj.disgust ? rio_disgust - optionObj.disgust : 0;
    if (angerInvolved) angerDiff = rio_anger > optionObj.anger ? rio_anger - optionObj.anger : 0;
    if (surpriseInvolved) {
      if (isPositiveSurprise(optionObj)) {
        surpriseDiff = optionObj.surprise > rio_surprise ? optionObj.surprise - rio_surprise : 0;
      } else {
        surpriseDiff = rio_surprise > optionObj.surprise ? rio_surprise - optionObj.surprise : 0;
      }
    }

    const diffArr = [sadnessDiff, joyDiff, fearDiff, disgustDiff, angerDiff, surpriseDiff];
    const labels = ['sadness', 'joy', 'fear', 'disgust', 'anger', 'surprise'];
    for (let i = 0; i < diffArr.length; i++) {
      if (diffArr[i] === undefined) {
        if (testMode) console.log(optionObj.label, "'s inclination is independent of ", labels[i]);
        diffArr.splice(i, 1); labels.splice(i, 1); i--;
      }
    }

    let weightedArray = weightsFor(diffArr.length);
    // ascending sort with labels
    for (let i = 0; i < diffArr.length - 1; i++) {
      for (let j = i + 1; j < diffArr.length; j++) {
        if (diffArr[i] > diffArr[j]) {
          const t = diffArr[i]; diffArr[i] = diffArr[j]; diffArr[j] = t;
          const t2 = labels[i]; labels[i] = labels[j]; labels[j] = t2;
        }
      }
    }

    let w = { sadness: 0, joy: 0, fear: 0, disgust: 0, anger: 0, surprise: 0 };
    for (let i = 0; i < diffArr.length; i++) {
      let v = diffArr[i] * weightedArray[i];
      switch (labels[i]) {
        case "sadness": w.sadness = (sadIncl !== undefined) ? (0.80 * v + 0.20 * sadIncl) : v; break;
        case "joy": w.joy = (joyIncl !== undefined) ? (0.80 * v + 0.20 * joyIncl) : v; break;
        case "fear": w.fear = (fearIncl !== undefined) ? (0.80 * v + 0.20 * fearIncl) : v; break;
        case "disgust": w.disgust = (disgustIncl !== undefined) ? (0.80 * v + 0.20 * disgustIncl) : v; break;
        case "anger": w.anger = (angerIncl !== undefined) ? (0.80 * v + 0.20 * angerIncl) : v; break;
        case "surprise": w.surprise = (surpriseIncl !== undefined) ? (0.80 * v + 0.20 * surpriseIncl) : v; break;
      }
    }

    let weightedAvg = w.sadness + w.joy + w.fear + w.disgust + w.anger + w.surprise;

    if (givenOption.hasOwnProperty("guilt_influence")) {
      let antiInclination = map(weightedAvg, 0, 1, 1, 0);
      antiInclination = antiInclination * currentAntiInclinationWeight + givenOption.guilt_influence * guiltInfluenceWeight;
      weightedAvg = map(antiInclination, 0, 1, 1, 0);
    }

    weightedAvg = clamp01(weightedAvg);
    return weightedAvg;
  }

  // pull basic emotion reference objects from memory (basic labels differ in testMode vs normal)
  const suf = testMode ? "__" : "_";
  const sadnessObj = memoryRet("longTerm", "sadness" + suf, false, null, false, false);
  const joyObj = memoryRet("longTerm", "joy" + suf, false, null, false, false);
  const fearObj = memoryRet("longTerm", "fear" + suf, false, null, false, false);
  const disgustObj = memoryRet("longTerm", "disgust" + suf, false, null, false, false);
  const angerObj = memoryRet("longTerm", "anger" + suf, false, null, false, false);
  const surpriseObj = memoryRet("longTerm", "surprise" + suf, false, null, false, false);

  const sadnessIncl = sadnessObj?.data ? calculateInclination(sadnessObj.data) : undefined;
  const joyIncl = joyObj?.data ? calculateInclination(joyObj.data) : undefined;
  const fearIncl = fearObj?.data ? calculateInclination(fearObj.data) : undefined;
  const disgustIncl = disgustObj?.data ? calculateInclination(disgustObj.data) : undefined;
  const angerIncl = angerObj?.data ? calculateInclination(angerObj.data) : undefined;
  const surpriseIncl = surpriseObj?.data ? calculateInclination(surpriseObj.data) : undefined;

  const total = calculateInclination(givenOption, sadnessIncl, joyIncl, fearIncl, disgustIncl, angerIncl, surpriseIncl);

  if (returnTheScore) return total;

  const repulsionValue = calculateRepulsion(givenOption);
  if (total > repulsionValue) return true;
  if (repulsionValue > total) return false;
  return "uncertain_state";
}

/* =============================
 * Prohibited choice handling & guilt stimulus
 * ============================= */
const otherObjects = []; // queued stimuli to be injected into emotion sphere

function generateGuiltStimulus(label, guilt_influence) {
  refreshSystemEmotions();
  const defaultGuiltSadness = rio_sadness;
  const defaultGuiltJoy = rio_joy;
  const maxPersonSadness = 1;
  const minPersonJoy = 0;
  let sigmaSadness = map(guilt_influence, 0, 1, 0, Math.abs(maxPersonSadness - defaultGuiltSadness));
  let sigmaJoy = map(guilt_influence, 0, 1, 0, Math.abs(defaultGuiltJoy - minPersonJoy));

  const changedSadness = clamp01(rio_sadness + sigmaSadness);
  const changedJoy = clamp01(rio_joy - sigmaJoy);
  const currentTime = Date.now();

  const guilt_stimulus = {
    label: label + "_guilt_stimulus_",
    anger: "system_anger_",
    joy: changedJoy,
    sadness: changedSadness,
    fear: "system_fear_",
    disgust: "system_disgust_",
    surprise: "system_surprise_",
    timesOccurred: 1,
    totalTimesUsed: 1,
    isProhibited: false,
    openness: Agent.bigFive.openness,
    conscientiousness: Agent.bigFive.conscientiousness,
    extraversion: Agent.bigFive.extraversion,
    agreeableness: Agent.bigFive.agreeableness,
    neuroticism: Agent.bigFive.neuroticism,
    relatedTasks: {},
    relatedStimuli: {},
    relatedPeople: {},
    time: currentTime,
    indexUnderLabel: 0
  };
  otherObjects.push(guilt_stimulus);
}

function chooseProhibitedObject(type, givenObject, speaker) {
  if (givenObject.isProhibited !== true) {
    console.error("chooseProhibitedObject: Given object is not prohibited.", givenObject);
    return false;
  }

  const speakerIdx = (speaker !== undefined) ? getPersonIndex(speaker) : -1;
  const personalitySim = (speakerIdx !== -1)
    ? PersonalityDiffScaledScore(personsPresent[speakerIdx])
    : PersonalityDiffScaledScore({
      personality: {
        openness: Agent.bigFive.openness,
        conscientiousness: Agent.bigFive.conscientiousness,
        extraversion: Agent.bigFive.extraversion,
        agreeableness: Agent.bigFive.agreeableness,
        neuroticism: Agent.bigFive.neuroticism
      }
    }, true);

  const likenessVal = (speakerIdx !== -1) ? personsPresent[speakerIdx].likeness : defaultLikeness;
  const timeEffectCoefficient = 1;
  const alpha = personalitySim * likenessVal * timeEffectCoefficient;

  // two-valued weighted average with abstentionConstant
  let weightedAlpha, weightedAbstention;
  if (alpha >= abstentionConstant) {
    weightedAbstention = abstentionConstant * weightsFor(2)[0];
    weightedAlpha = alpha * weightsFor(2)[1];
  } else {
    weightedAbstention = abstentionConstant * weightsFor(2)[1];
    weightedAlpha = alpha * weightsFor(2)[0];
  }
  const change = weightedAlpha + weightedAbstention;

  // Save/update prohibited object (with guilt_influence baked)
  const relatedStimuli = Object.keys(givenObject.relatedStimuli || {});
  const relatedPeople = Object.keys(givenObject.relatedPeople || {});
  const relatedEmotions = [
    givenObject.sadness, givenObject.joy, givenObject.fear,
    givenObject.disgust, givenObject.anger, givenObject.surprise
  ];
  const idx = memoryRet("longTerm", givenObject.label, false, null, false, true, relatedStimuli, relatedPeople, relatedEmotions)?.index;
  const objCopy = Object.assign({}, givenObject, { guilt_influence: change });
  memoryRet("longTerm", givenObject.label, true, objCopy, false, true, null, null, null, idx);

  // Build guilt stimulus now (and queue it to inject)
  if (type === "verbal") {
    // if the prohibited thing is a verbal object, we use its label for guilt stimulus
    generateGuiltStimulus(givenObject.label, change);
  } else {
    generateGuiltStimulus(givenObject.label, change);
  }

  // Decide if chosen: compare inclination(prohibited) vs repulsion(guilt)
  const prohibitedObjMinimal = {
    label: givenObject.label,
    sadness: givenObject.sadness,
    joy: givenObject.joy,
    fear: givenObject.fear,
    disgust: givenObject.disgust,
    anger: givenObject.anger,
    surprise: givenObject.surprise
  };

  const choose = (hasSufficientEmWeight(prohibitedObjMinimal, true) > calculateRepulsion(otherObjects[otherObjects.length - 1]));
  if (!choose) {
    // we *did not* choose the prohibited thing → guilt stimulus stays queued but option rejected
    return false;
  }
  // we *did* choose the prohibited option; guilt stimulus will be injected to emotion sphere.
  return true;
}

/* =============================
 * Emotional Decide (full)
 * ============================= */
let lastChosenOption = null;

function emotionalDecide(action, options, speaker) {
  console.log("Emotional decision requested:", { action, options, speaker });

  if (action === undefined) action = "chooseOption_";
  if (!Array.isArray(options)) options = [];

  let decided = false;
  const decision = { type: null, subtype: null, object: null };

  // ACTION CHECK
  if (!decided) {
    const actionObj = memoryRet("longTerm", action, false, null, false, false);
    if (!actionObj) {
      decision.type = "action_unknown";
      decided = true;
    }
    else {

      // calculate the choice score based on emotions and personality

      const emotionalSore = hasSufficientEmWeight(actionObj.data, true) * inclinationWeight;
      const personalityScore = personalityNearby(actionObj.data, true) * personalitySimilarityWeight;
      const repulsionScore = calculateRepulsion(actionObj.data, true) * repulsionWeight;
      const choiceScore = emotionalSore + personalityScore - repulsionScore;

      console.log('emotionalWeight:', emotionalSore, 'personalityWeight:', personalityScore, 'repulsionWeight:', repulsionScore);

      // const choiceScore =
      //   hasSufficientEmWeight(actionObj.data, true) * inclinationWeight +
      //   personalityNearby(actionObj.data, true) * personalitySimilarityWeight +
      //   calculateRepulsion(actionObj.data, true) * repulsionWeight;

      console.log('choice score:', choiceScore, 'for action:', actionObj.data.label);

      const actionLiked = (choiceScore >= choiceThreshold);
      if (options.length === 0) {
        decision.type = actionLiked ? "action_liked" : "action_not_liked";
        decision.object = actionObj.data;

        if (decision.object.isProhibited) {
          if (chooseProhibitedObject("choice", decision.object, speaker) === false) {
            decision.type = "action_not_liked";
            decision.object = null;
          }
        }
        decided = true;
      } else {
        if (!actionLiked) {
          decision.type = "action_not_liked";
          decision.object = null;
          decided = true;
        }
      }
    }
  }

  // OPTION CHECK
  if (!decided) {
    if (options.length === 1) {
      decision.type = "single_option_type";
      const optionObj = memoryRet("longTerm", options[0], false, null, false, false);
      if (!optionObj) {
        decision.subtype = "option_unknown";
        decided = true;
      } else {
        const choiceScore =
          hasSufficientEmWeight(optionObj.data, true) * inclinationWeight +
          personalityNearby(optionObj.data, true) * personalitySimilarityWeight +
          calculateRepulsion(optionObj.data, true) * repulsionWeight;

        const optionLiked = (choiceScore >= choiceThreshold);
        decision.subtype = optionLiked ? "option_liked" : "option_not_selected";
        if (decision.subtype === "option_liked") {
          decision.object = optionObj.data;
          if (decision.object.isProhibited) {
            if (chooseProhibitedObject("choice", decision.object, speaker) === false) {
              decision.subtype = "option_not_selected";
              decision.object = null;
            }
          }
        }
        decided = true;
      }
    } else {
      // MULTI OPTION
      decision.type = "multi_option_type";
      const knownOptions = [];
      for (const lab of options) {
        const obj = memoryRet("longTerm", lab, false, null, false, false);
        if (obj) knownOptions.push(obj.data);
      }
      if (knownOptions.length === 0) {
        decision.subtype = "options_unknown";
        decided = true;
      } else if (knownOptions.length === 1) {
        const choiceScore =
          hasSufficientEmWeight(knownOptions[0], true) * inclinationWeight +
          personalityNearby(knownOptions[0], true) * personalitySimilarityWeight +
          calculateRepulsion(knownOptions[0], true) * repulsionWeight;

        const optionLiked = (choiceScore >= choiceThreshold);
        decision.subtype = optionLiked ? knownOptions[0].label : "options_not_selected";
        if (decision.subtype !== "options_not_selected") {
          decision.object = knownOptions[0];
          if (decision.object.isProhibited) {
            if (chooseProhibitedObject("choice", decision.object, speaker) === false) {
              decision.subtype = "options_not_selected";
              decision.object = null;
            }
          }
        }
        decided = true;
      } else {
        const uncertainThreshold = 0.015;
        let weightedOptions = [];
        for (const el of knownOptions) {
          const emScore = hasSufficientEmWeight(el, true);
          const choiceScore =
            personalityNearby(el, true) * personalitySimilarityWeight +
            emScore * inclinationWeight +
            calculateRepulsion(el, true) * repulsionWeight;
          if (choiceScore >= choiceThreshold) weightedOptions.push({ label: el.label, score: emScore });
        }
        weightedOptions = bubblSrt(weightedOptions, "score");
        let chosen_option;

        if (weightedOptions.length === 0) {
          chosen_option = "options_not_selected";
        } else {
          const first = weightedOptions[weightedOptions.length - 1].score;
          if (weightedOptions.length >= 2) {
            const second = weightedOptions[weightedOptions.length - 2].score;
            if (Math.abs(first - second) < uncertainThreshold) chosen_option = "uncertain_state";
          }
          if (chosen_option !== "uncertain_state") {
            // walk from highest down; avoid prohibited unless guilt choice keeps it
            let picked = null;
            for (let i = weightedOptions.length - 1; i >= 0; i--) {
              const lab = weightedOptions[i].label;
              const found = knownOptions.find(k => k.label === lab);
              if (!found) continue;
              if (found.isProhibited) {
                if (chooseProhibitedObject("choice", found, speaker) && weightedOptions[i].score >= choiceThreshold) {
                  picked = found; break;
                }
              } else {
                picked = found; break;
              }
            }
            if (picked) {
              chosen_option = picked.label;
              decision.object = Object.assign({}, picked);
            } else {
              chosen_option = "options_not_selected";
            }
          }
        }
        decision.subtype = chosen_option;
        decided = true;
      }
    }
  }

  if (decided) {
    if (decision.object) {
      // anticipated stimulus (unless special)
      if (!checkIfSpecialDecisions(decision.object.label)) {
        const anticipated = Object.assign({}, decision.object, {
          label: decision.object.label + "_anticipated_stimulus_",
          time: Date.now()
        });
        otherObjects.push(anticipated);
      }
      lastChosenOption = Object.assign({}, decision.object);
    }
    return decision;
  }
  console.error("Error: emotionalDecide() not decided.");
  return { type: "action_unknown", subtype: null, object: null };
}

/* =============================
 * Stimulus Creation + Injection
 * ============================= */
function makeStimulus(label, emo, traits = {}, opts = {}) {
  return {
    label,
    sadness: clamp01(emo.sadness || 0),
    joy: clamp01(emo.joy || 0),
    fear: clamp01(emo.fear || 0),
    disgust: clamp01(emo.disgust || 0),
    anger: clamp01(emo.anger || 0),
    surprise: clamp01(emo.surprise || 0),
    timesOccurred: 1,
    totalTimesUsed: 1,
    isProhibited: !!opts.isProhibited,
    openness: clamp01(traits.openness ?? Agent.bigFive.openness),
    conscientiousness: clamp01(traits.conscientiousness ?? Agent.bigFive.conscientiousness),
    extraversion: clamp01(traits.extraversion ?? Agent.bigFive.extraversion),
    agreeableness: clamp01(traits.agreeableness ?? Agent.bigFive.agreeableness),
    neuroticism: clamp01(traits.neuroticism ?? Agent.bigFive.neuroticism),
    relatedTasks: {},
    relatedStimuli: {},
    relatedPeople: {},
    time: Date.now(),
    indexUnderLabel: 0
  };
}
function injectQueuedStimuli() {
  if (!otherObjects.length) return;
  const objs = otherObjects.splice(0, otherObjects.length);
  emotion.input({
    type: "other",
    speaker: "system",
    likeness: 0.5,
    personality: Agent.bigFive,
    objects: objs
  });
}

/* =============================
 * Main Response (verbal; includes decision flow)
 * ============================= */
const terminate_Word = "goodbye"; // simple terminator

async function mainResponse(givenText, currentSpeaker = "user") {
  let responseText = "";
  const decisionObjects = decisionSpeechExtractor.decisionSpeech(givenText);

  if (decisionObjects != null) {
    var DecisionmainVerb = decisionObjects.verbs[0];     // keep the original for speaking later
    var decisionOptions = decisionObjects.objects || []; // originals for speaking

    // --- NEW: stem for decision logic, keep originals for response text ---
    const DecisionmainVerbStem = stemmer.stem(String(DecisionmainVerb || "").toLowerCase());
    const decisionOptionsStem = decisionOptions.map(o => stemmer.stem(String(o || "").toLowerCase()));

    // map stems back to original surfaces so we never speak stems
    const surfaceByStem = new Map();
    surfaceByStem.set(DecisionmainVerbStem, DecisionmainVerb);
    for (let i = 0; i < decisionOptions.length; i++) {
      surfaceByStem.set(decisionOptionsStem[i], decisionOptions[i]);
    }
    const surfaceForStem = (s) => surfaceByStem.get(s) || s; // fallback to stem if unseen

    // decide using STEMS only
    var decision = emotionalDecide(DecisionmainVerbStem, decisionOptionsStem, currentSpeaker);
    const randomNum = Math.round(Math.random() * 4);
    switch (decision.type) {
      case "action_liked":
        switch (randomNum) {
          case 1: responseText += "I will prefer "; break;
          case 2: responseText += "I'll go for "; break;
          case 3: responseText += "I'll choose "; break;
          default: responseText += "I'll favour ";
        }
        if (DecisionmainVerb.endsWith("ing")) responseText += DecisionmainVerb + ".";
        else responseText += "to " + DecisionmainVerb + ".";
        break;

      case "action_not_liked":
        switch (randomNum) {
          case 1: responseText += "I don't prefer "; break;
          case 2: responseText += "I don't like "; break;
          case 3: responseText += "I will not choose "; break;
          default: responseText += "I don't favour ";
        }
        if (DecisionmainVerb.endsWith("ing")) responseText += DecisionmainVerb + ".";
        else responseText += "to " + DecisionmainVerb + ".";
        break;

      case "action_unknown":
        if (DecisionmainVerb.endsWith("ing")) {
          switch (randomNum) {
            case 1: responseText += "I do not have any idea about " + DecisionmainVerb + "."; break;
            case 2: responseText += "I have no idea what " + DecisionmainVerb + " is."; break;
            default: responseText += "The meaning of " + DecisionmainVerb + " is not known by me.";
          }
        } else {
          switch (randomNum) {
            case 1: responseText += "I don't know what " + DecisionmainVerb + " means."; break;
            case 2: responseText += "I don't have any idea how to " + DecisionmainVerb + "."; break;
            default: responseText += "I have no idea what " + DecisionmainVerb + " implies.";
          }
        }
        break;

      case "single_option_type":
        if (decision.subtype === "option_liked") {
          const opt = (decisionOptions[0] || "").trim();
          switch (randomNum) {
            case 1: responseText += "I will surely go for " + opt + "."; break;
            case 2: responseText += "I'll favour " + opt + "."; break;
            case 3: responseText += "I'll surely choose " + opt + "."; break;
            default: responseText += "I will prefer " + opt + ".";
          }
        } else if (decision.subtype === "option_unknown") {
          const opt = (decisionOptions[0] || "").trim();
          switch (randomNum) {
            case 1: responseText += "I don't have any idea about " + opt + "."; break;
            case 2: responseText += "I haven't got any idea about " + opt + "."; break;
            default: responseText += "I have not learnt what " + opt + " is.";
          }
        } else if (decision.subtype === "option_not_selected") {
          const opt = (decisionOptions[0] || "").trim();
          switch (randomNum) {
            case 1: responseText += "I will never go for " + opt + "."; break;
            case 2: responseText += "I will not prefer " + opt + "."; break;
            case 3: responseText += "I will not select " + opt + "."; break;
            default: responseText += "I won't like " + opt + ".";
          }
        } else {
          console.error("Invalid decision subtype for single_option_type.");
        }
        break;

      case "multi_option_type":
        if (decision.subtype === "options_unknown") {
          switch (randomNum) {
            case 1: responseText += "I don't have any idea about any of the options given."; break;
            case 2: responseText += "I have not learnt about any of the options I have."; break;
            default: responseText += "I know nothing about the options I have.";
          }
        } else if (decision.subtype === "uncertain_state") {
          switch (randomNum) {
            case 1: responseText += "I am totally confused. I don't know what to choose."; break;
            case 2: responseText += "I am uncertain and puzzled. I don't know what to choose."; break;
            default: responseText += "I don't know what to choose. I am totally puzzled.";
          }
        } else if (decision.subtype === "options_not_selected") {
          let speakverb = DecisionmainVerb.endsWith("ing") ? DecisionmainVerb : "to " + DecisionmainVerb;
          switch (randomNum) {
            case 1: responseText += "I prefer " + speakverb + ". But I don't like any of the options."; break;
            case 2: responseText += "I will favour " + speakverb + ". But I don't favour my options."; break;
            default: responseText += "I will surely like " + speakverb + ". However I don't prefer my options.";
          }
        }

        // multi_option_type ... // when we have actually selected something
        else {
          const chosenSurface = surfaceForStem(decision.subtype); // <-- map stem → original
          switch (randomNum) {
            case 1: responseText += "I will favour " + chosenSurface + ".";
              break;
            case 2: responseText += "I will surely go for " + chosenSurface + ".";
              break;
            case 3: responseText += "I will select " + chosenSurface + ".";
              break;
            default: responseText += "I will prefer " + chosenSurface + ".";
          }
        }

        break;

      default:
        responseText += "";
    }
  }
  else if (!givenText.toLowerCase().includes(terminate_Word)) {
    // general speech (not terminating)
    // We accumulate text; NLU-driven stimuli and final echo handled elsewhere
    responseText += ""; // keep minimal; use emotion-aware echo in /output
  } else {
    console.error("Error: mainResponse(): decisionObjects undefined but terminating state hit.");
  }
  return responseText;
}

/* =============================
 * Express Server (frontend compatibility)
 * ============================= */
const app = express();
app.use(bodyParser.json({ limit: "1mb" }));

function startOnce() {
  if (Agent.started) return;
  emotion.setPersonalityTraits(Agent.bigFive);
  emotion.emotion();
  Agent.started = true;
  console.log("✅ Emotional engine online.");
}

// INIT handshake
let displayedOnce = false;
app.post("/initialisation", (req, res) => {
  if (req.body?.module === "frontend") {
    if (!displayedOnce) {
      app.post("/startTheBackend", (req2, res2) => {
        startOnce();
        res2.json({ start: true });
      });
      displayedOnce = true;
    }
    return res.json({ start: true });
  }
  res.json({ start: false });
});

// OUTPUT: Frontend → Backend
app.post("/output", async (req, res) => {
  try {
    const b = req.body || {};
    if (b.recognisedSpeech?.data) {
      const raw = String(b.recognisedSpeech.data);
      const text = raw.startsWith(" ") ? raw.slice(1) : raw;
      const msgLower = text.toLowerCase();

      // basic prohibited speech scanning
      const verbalCant = /(?:don't speak|do not speak|never say|don't say)/i.test(msgLower);

      // speaker setup
      const speaker = "user";
      const speakerIdx = ensureSpeaker(speaker);
      if (msgLower.includes(Agent.name.toLowerCase())) {
        personsPresent[speakerIdx].likeness = clamp01(personsPresent[speakerIdx].likeness + 0.05);
      }

      // Decision speech → response (speak/not)
      const agentEmo = emotion.getEmotions();
      const reply = await mainResponse(text, speaker);
      let willSpeak = !!reply && !verbalCant;
      if (willSpeak) queueSpeech += reply + " ";

      // Face / voice hints
      const face = faceFromEmotions(agentEmo);
      F.frontend_data.faceExpression = face.face;
      F.frontend_data.pitch = face.pitch;
      F.frontend_data.rate = face.rate;
      F.frontend_data.speakingState = willSpeak;


      // Tokenize + stem with natural → per-token stimuli
      const stimuli = await tokenStimuliFromText(text, personsPresent[speakerIdx].personality);

      // Fallback: if everything got filtered out, keep legacy one-shot behavior once
      if (stimuli.length === 0) {
        const nlu = await analyzeNLU(text);
        const fallbackLabel = hashLabel("verbal", text);
        stimuli.push(makeStimulus(fallbackLabel, nlu.emotion, personsPresent[speakerIdx].personality, { isProhibited: false }));
      }

      // inject all token stimuli at once
      emotion.input({
        type: "verbal",
        speaker,
        likeness: personsPresent[speakerIdx].likeness,
        personality: personsPresent[speakerIdx].personality,
        objects: stimuli
      });

      // persist each token stimulus exactly as learned (by stem label)
      for (const s of stimuli) {
        memoryRet("longTerm", s.label, true, Object.assign({}, s), true, true);
      }

      // Inject any anticipated/guilt stimuli queued by decisions
      injectQueuedStimuli();
    }

    res.json({ received: true });
  } catch (e) {
    console.error("Error in /output:", e);
    res.status(500).json({ error: String(e) });
  }
});

// INPUT: Backend → Frontend (polling)
app.post("/input", (req, res) => {
  if (queueSpeech.length) {
    F.frontend_data.speech = (F.frontend_data.speech || "") + queueSpeech;
    queueSpeech = "";
  }

  const em = emotion.getEmotions();
  const maxVal = Math.max(em.sadness, em.joy, em.fear, em.disgust, em.anger, em.surprise);
  F.frontend_data.emotion = em;
  F.frontend_data.emotionScore = maxVal;

  res.json(F);

  // clear one-shots
  ["audio_incoming", "song_incoming", "speaker_incoming", "visual_incoming", "song_incomingJS"].forEach(k => {
    if (F[k].text !== null) F[k].text = null;
    if (F[k].val !== null) F[k].val = null;
    if (F[k].json !== null) F[k].json = null;
  });
  if (F.verbal_incoming.text !== null) F.verbal_incoming.text = null;
  if (F.verbal_incoming.val !== null) F.verbal_incoming.val = null;
  if (F.verbal_incoming.json.length) F.verbal_incoming.json.length = 0;
  if (F.pose_incoming.text !== null) F.pose_incoming.text = null;
  if (F.pose_incoming.val !== null) F.pose_incoming.val = null;
  if (F.pose_incoming.json.length) F.pose_incoming.json.length = 0;
  if (F.frontend_data.speech !== null) F.frontend_data.speech = null;
});

// SAVE/RESTORE (persist learned stimuli exactly as stored)
app.post("/restore", (req, res) => {
  const mem = readMemory();
  res.json({
    verbal_restore: { dataset: mem.longTerm || [] },
    pose_restore: { dataset: null },
    altPoseRestore: { dataset: null },
    audio_restore: { dataset: null },
    song_restore: { dataset: null },
    speaker_restore: { dataset: null },
    visual_restore: { dataset: null }
  });
});

app.post("/save", (req, res) => {
  // Accept incoming datasets (if any) and persist.
  // If frontends call /save with learned objects, merge them.
  try {
    const mem = readMemory();
    const body = req.body || {};
    const inVerbal = body?.verbal_restore?.dataset;
    if (Array.isArray(inVerbal)) {
      // merge by label (prefer incoming)
      const mapOld = new Map((mem.longTerm || []).map(o => [o.label, o]));
      for (const obj of inVerbal) mapOld.set(obj.label, obj);
      mem.longTerm = Array.from(mapOld.values());
      writeMemory(mem);
    }
    res.json({ saved: true });
  } catch (e) {
    console.error("save error:", e);
    res.status(500).json({ saved: false, error: String(e) });
  }
});


// POST /learn_stimulus
// body: { label, emotions:{sadness,joy,fear,disgust,anger,surprise}, isProhibited?, traits?, inject? }
// returns: { ok:true, stimulus }
app.post("/learn_stimulus", (req, res) => {
  const { label, emotions = {}, isProhibited = false, traits = {}, inject = false } = req.body || {};
  if (!label) return res.status(400).json({ ok: false, error: "label required" });

  const stim = makeStimulus(label, emotions, traits, { isProhibited });
  // persist exactly as learned
  memoryRet("longTerm", stim.label, true, Object.assign({}, stim), true, true);

  if (inject) {
    emotion.input({
      type: "other",
      speaker: "teacher",
      likeness: 0.5,
      personality: traits || {},
      objects: [stim]
    });
  }
  res.json({ ok: true, stimulus: stim });
});


// POST /mark_prohibited
// body: { label, prohibited: boolean, guilt_influence? }
// returns: { ok:true, updated }
app.post("/mark_prohibited", (req, res) => {
  const { label, prohibited, guilt_influence } = req.body || {};
  if (!label || typeof prohibited !== "boolean")
    return res.status(400).json({ ok: false, error: "label and prohibited(boolean) required" });

  const rec = memoryRet("longTerm", label, false, null, false, false);
  if (!rec) return res.status(404).json({ ok: false, error: "not found" });

  const updated = Object.assign({}, rec.data, {
    isProhibited: !!prohibited
  });
  if (typeof guilt_influence === "number") updated.guilt_influence = clamp01(guilt_influence);

  memoryRet("longTerm", label, true, updated, false, true);

  // Optionally create an immediate guilt stimulus if you just prohibited it with a known influence:
  if (updated.isProhibited && typeof updated.guilt_influence === "number") {
    generateGuiltStimulus(label, updated.guilt_influence);
    injectQueuedStimuli();
  }

  // send this back
  res.json({ ok: true, updated });
});


// inspection of memory stimuli
// GET /memory/stimuli?q=partial
// returns: array of {label,...}
app.get("/memory/stimuli", (req, res) => {
  const q = (req.query.q || "").toString().toLowerCase();
  const mem = readMemory();
  const all = Array.isArray(mem.longTerm) ? mem.longTerm : [];
  const out = q ? all.filter(o => (o.label || "").toLowerCase().includes(q)) : all;
  res.json(out);
});

// GET /memory/stimuli/:label
app.get("/memory/stimuli/:label", (req, res) => {
  const label = req.params.label;
  const rec = memoryRet("longTerm", label, false, null, false, false);
  if (!rec) return res.status(404).json({ ok: false, error: "not found" });
  res.json(rec.data);
});

// DELETE /memory/stimuli/:label
app.delete("/memory/stimuli/:label", (req, res) => {
  const ok = removeFromMemory(req.params.label);
  res.json({ ok });
});


// External emotional decision
// POST /decide
// body: { action, options: [labels], speaker }
// returns: full decision object
app.post("/decide", (req, res) => {
  const { action, options, speaker } = req.body || {};
  const d = emotionalDecide(action, options, speaker);
  // decisions may have queued stimuli (anticipated/guilt) → inject now
  injectQueuedStimuli();
  res.json(d);
});


// inject arbitrary stimulus at any time
// POST /stimulate
// body: { label, emotions:{...}, traits?, type?="other", speaker?="system", likeness?=0.5, isProhibited? }
// returns: { ok:true, stimulus }
app.post("/stimulate", (req, res) => {
  const { label, emotions = {}, traits = {}, type = "other", speaker = "system", likeness = 0.5, isProhibited = false } = req.body || {};
  if (!label) return res.status(400).json({ ok: false, error: "label required" });

  const stim = makeStimulus(label, emotions, traits, { isProhibited });
  emotion.input({ type, speaker, likeness: clamp01(likeness), personality: traits || {}, objects: [stim] });
  memoryRet("longTerm", stim.label, true, Object.assign({}, stim), true, true);

  res.json({ ok: true, stimulus: stim });
});

// speaker management to create/update a speaker profile the decision system can use
// PUT /speaker
// body: { name, likeness?, personality?{openness, conscientiousness, extraversion, agreeableness, neuroticism} }
// returns: { ok:true, speaker }
app.put("/speaker", (req, res) => {
  const { name, likeness, personality } = req.body || {};
  if (!name) return res.status(400).json({ ok: false, error: "name required" });
  const idx = ensureSpeaker(name);
  if (typeof likeness === "number") personsPresent[idx].likeness = clamp01(likeness);
  if (personality) personsPresent[idx].personality = Object.assign({}, personsPresent[idx].personality, personality);
  res.json({ ok: true, speaker: personsPresent[idx] });
});


// current emotional ststae snapshot for dashboards
// GET /state
// returns: { emotions, lastChosenOption }
app.get("/state", (req, res) => {
  res.json({ emotions: emotion.getEmotions(), lastChosenOption });
});


// manually generate a guilt stimulus
// POST /guilt
// body: { label, guilt_influence }
// returns: { ok:true }
app.post("/guilt", (req, res) => {
  const { label, guilt_influence } = req.body || {};
  if (!label || typeof guilt_influence !== "number")
    return res.status(400).json({ ok: false, error: "label and numeric guilt_influence required" });
  generateGuiltStimulus(label, clamp01(guilt_influence));
  injectQueuedStimuli();
  res.json({ ok: true });
});


app.use(express.static(path.join(__dirname, "public")));
app.get("/dashboard", (_, res) => res.sendFile(path.join(__dirname, "public", "dashboard.html")));


/* =============================
 * Run Server
 * ============================= */
app.listen(PORT, () => {
  console.log(`Conscious emotional backend (full) on http://127.0.0.1:${PORT}`);
  // start immediately (your frontends also call /startTheBackend, which is idempotent)
  startOnce();
});
