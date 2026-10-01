// extension/js/inference.js
// Character-level seq2seq autocorrect inference in plain JavaScript - no
// TensorFlow.js, no WebGL, no multi-megabyte runtime. The whole model is a
// pair of small LSTMs (~0.2M parameters); running the forward pass by hand is
// a few matrix-vector products and comfortably real-time per word.
//
// This mirrors ai_model/src/inference.py exactly, including the Keras LSTM gate
// order (input, forget, cell, output) and the dictionary gate: a known word is
// never changed, and a proposal is only used if it is itself a real word.
'use strict';

(function (global) {
  function b64ToFloat32(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  class Corrector {
    // weights: object of base64 Float32 tensors; meta: {U, V, EMB}; tok: tokenizer
    // config; dictionary: Set<string>.
    constructor(weights, meta, tok, dictionary) {
      const f = (k) => b64ToFloat32(weights[k]);
      this.encEmb = f('enc_emb'); this.encK = f('enc_k'); this.encRk = f('enc_rk'); this.encB = f('enc_b');
      this.decEmb = f('dec_emb'); this.decK = f('dec_k'); this.decRk = f('dec_rk'); this.decB = f('dec_b');
      this.denK = f('den_k'); this.denB = f('den_b');
      this.U = meta.U; this.V = meta.V; this.EMB = meta.EMB;

      this.c2i = tok.char_to_index;
      this.i2c = {};
      for (const [k, v] of Object.entries(tok.index_to_char)) this.i2c[parseInt(k, 10)] = v;
      this.maxlen = tok.max_seq_length;
      this.start = tok.start_token_index;
      this.end = tok.end_token_index;
      this.pad = tok.pad_token_index;
      this.dictionary = dictionary;
    }

    // One LSTM step. x: Float32Array[EMB] (an embedding row). kernel K is
    // [EMB, 4U] row-major; recurrent RK is [U, 4U]; bias B is [4U].
    _step(x, h, c, K, RK, B) {
      const U = this.U, four = 4 * U;
      const z = new Float32Array(four);
      z.set(B);
      for (let i = 0; i < x.length; i++) {
        const xi = x[i], base = i * four;
        for (let j = 0; j < four; j++) z[j] += xi * K[base + j];
      }
      for (let i = 0; i < U; i++) {
        const hi = h[i], base = i * four;
        for (let j = 0; j < four; j++) z[j] += hi * RK[base + j];
      }
      const nh = new Float32Array(U), nc = new Float32Array(U);
      for (let u = 0; u < U; u++) {
        const ig = sigmoid(z[u]);
        const fg = sigmoid(z[U + u]);
        const g = Math.tanh(z[2 * U + u]);
        const og = sigmoid(z[3 * U + u]);
        nc[u] = fg * c[u] + ig * g;
        nh[u] = og * Math.tanh(nc[u]);
      }
      return [nh, nc];
    }

    _embRow(table, idx) {
      return table.subarray(idx * this.EMB, (idx + 1) * this.EMB);
    }

    _encode(word) {
      let h = new Float32Array(this.U), c = new Float32Array(this.U);
      const seq = [this.start];
      for (const ch of word) seq.push(this.c2i[ch] !== undefined ? this.c2i[ch] : this.pad);
      seq.push(this.end);
      for (const t of seq) [h, c] = this._step(this._embRow(this.encEmb, t), h, c, this.encK, this.encRk, this.encB);
      return [h, c];
    }

    _logits(h) {
      const V = this.V, U = this.U, out = new Float32Array(V);
      for (let v = 0; v < V; v++) {
        let s = this.denB[v];
        for (let u = 0; u < U; u++) s += h[u] * this.denK[u * V + v];
        out[v] = s;
      }
      return out;
    }

    _logSoftmax(v) {
      let m = -Infinity;
      for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
      let sum = 0;
      for (let i = 0; i < v.length; i++) sum += Math.exp(v[i] - m);
      const ls = Math.log(sum) + m;
      const out = new Float32Array(v.length);
      for (let i = 0; i < v.length; i++) out[i] = v[i] - ls;
      return out;
    }

    // Beam search: returns likely corrections, best first. Mirrors
    // inference.py exactly so Python and the browser agree.
    candidates(word, beamWidth = 5, maxCandidates = 8) {
      let [h0, c0] = this._encode(word.toLowerCase());
      let beams = [{ score: 0, seq: [this.start], h: h0, c: c0 }];
      const finished = [];
      for (let k = 0; k < this.maxlen; k++) {
        const nxt = [];
        for (const b of beams) {
          const [nh, nc] = this._step(this._embRow(this.decEmb, b.seq[b.seq.length - 1]), b.h, b.c, this.decK, this.decRk, this.decB);
          const logp = this._logSoftmax(this._logits(nh));
          const idx = Array.from(logp.keys()).sort((a, z) => logp[z] - logp[a]).slice(0, beamWidth);
          for (const t of idx) {
            if (t === this.end || t === this.pad) finished.push({ score: b.score + logp[t], seq: b.seq });
            else nxt.push({ score: b.score + logp[t], seq: b.seq.concat(t), h: nh, c: nc });
          }
        }
        if (!nxt.length) break;
        nxt.sort((a, z) => (z.score / Math.max(z.seq.length, 1)) - (a.score / Math.max(a.seq.length, 1)));
        beams = nxt.slice(0, beamWidth);
      }
      const pool = finished.concat(beams);
      pool.sort((a, z) => (z.score / Math.max(z.seq.length, 1)) - (a.score / Math.max(a.seq.length, 1)));
      const words = [], seen = new Set();
      for (const b of pool) {
        let txt = '';
        for (const t of b.seq) {
          if (t === this.start || t === this.end || t === this.pad) continue;
          const ch = this.i2c[t];
          if (ch && ch !== '\t' && ch !== '\n') txt += ch;
        }
        if (txt && !seen.has(txt)) { seen.add(txt); words.push(txt); }
        if (words.length >= maxCandidates) break;
      }
      return words;
    }

    correct(word) {
      const lower = word.toLowerCase();
      if (!/^[a-z]+$/.test(lower) || this.dictionary.has(lower)) return word;
      for (const cand of this.candidates(lower)) {
        if (cand !== lower && this.dictionary.has(cand)) return cand;
      }
      return word;
    }
  }

  async function load(basePath) {
    const [weights, tok, dictText] = await Promise.all([
      fetch(basePath + 'weights.json').then((r) => r.json()),
      fetch(basePath + 'tokenizer_config.json').then((r) => r.json()),
      fetch(basePath + 'dictionary.txt').then((r) => r.text()),
    ]);
    const dictionary = new Set(dictText.split('\n').map((s) => s.trim()).filter(Boolean));
    return new Corrector(weights.tensors, weights.meta, tok, dictionary);
  }

  global.GTCInference = { Corrector, load };
})(typeof self !== 'undefined' ? self : this);

// Node (tests) can require this file.
if (typeof module !== 'undefined' && module.exports) {
  module.exports = (typeof self !== 'undefined' ? self : this).GTCInference;
}
