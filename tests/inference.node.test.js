// Node test: the browser inference engine (extension/js/inference.js) must
// produce exactly the corrections the Python reference does. Run: node this file.
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const MODEL = path.join(ROOT, 'extension', 'model');
const GTC = require(path.join(ROOT, 'extension', 'js', 'inference.js'));

const weights = JSON.parse(fs.readFileSync(path.join(MODEL, 'weights.json')));
const tok = JSON.parse(fs.readFileSync(path.join(MODEL, 'tokenizer_config.json')));
const dict = new Set(
  fs.readFileSync(path.join(MODEL, 'dictionary.txt'), 'utf8').split('\n').map((s) => s.trim()).filter(Boolean)
);
const c = new GTC.Corrector(weights.tensors, weights.meta, tok, dict);

// Same expectations as tests/test_inference.py::test_fixes_common_typos.
const expected = {
  teh: 'the', recieve: 'receive', beleive: 'believe',
  wrld: 'world', thsi: 'this', langauge: 'language',
};
for (const [typo, want] of Object.entries(expected)) {
  assert.strictEqual(c.correct(typo), want, `correct(${typo})`);
}

// Known words are never changed.
for (const w of ['the', 'hello', 'world', 'language', 'receive']) {
  assert.strictEqual(c.correct(w), w, `kept(${w})`);
}

// Output is always the input or a real dictionary word.
for (const junk of ['xqzptk', 'zzzz', 'qwxrt']) {
  const out = c.correct(junk);
  assert.ok(out === junk || dict.has(out), `only-real-words(${junk})`);
}

// Non-alphabetic tokens are left alone.
for (const t of ['', '123', "don't"]) {
  assert.strictEqual(c.correct(t), t, `left-alone(${t})`);
}

console.log('inference.node.test.js: all assertions passed');
