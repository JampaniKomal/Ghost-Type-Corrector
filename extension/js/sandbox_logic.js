// extension/js/sandbox_logic.js
// Runs inside the extension's sandboxed iframe. Loads the autocorrect model
// (plain-JS inference, see inference.js) and answers prediction requests from
// the content script. No network, no TensorFlow.js runtime.
'use strict';

let corrector = null;

async function initialize(modelBase) {
  try {
    corrector = await GTCInference.load(modelBase);
    window.parent.postMessage({ type: 'GTC_READY' }, '*');
    console.log('Sandbox: model loaded, ready');
  } catch (error) {
    console.error('Sandbox: initialization failed:', error);
    window.parent.postMessage(
      { type: 'GTC_ERROR', message: `AI initialization failed: ${error.message}` }, '*');
  }
}

window.addEventListener('message', (event) => {
  if (event.source !== window.parent || !event.data) return;

  if (event.data.type === 'GTC_INIT') {
    initialize(event.data.modelBase);
  } else if (event.data.type === 'GTC_PREDICT') {
    const word = event.data.word;
    if (!corrector || typeof word !== 'string') return;
    const corrected = corrector.correct(word);
    window.parent.postMessage(
      { type: 'GTC_RESULT', originalWord: word, correctedWord: corrected }, '*');
  }
});

console.log('Sandbox: script loaded, waiting for initialization message...');
