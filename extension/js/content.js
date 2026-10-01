// extension/js/content.js
// Main script for Ghost Type Corrector: Sets up the sandboxed AI environment 
// and handles communication, user input, and text replacement.

(function() {
    'use strict';
    console.log("Ghost Type Corrector: Content script loaded (Sandbox Mode)");

    // --- Global Variables ---
    let sandboxReady = false;
    let sandboxIframe = null; 

    // Variables for Text Correction
    let lastOriginalWord = null;
    let lastCorrectedWord = null;
    let lastProcessedElement = null;

    // --- Helper Function: Case Matching ---
    function matchCase(originalWord, correctedWord) {
        if (!originalWord || !correctedWord) return correctedWord;

        // All caps
        if (originalWord === originalWord.toUpperCase()) {
            return correctedWord.toUpperCase();
        }
        
        // Title case (first letter capitalized)
        if (originalWord[0] === originalWord[0].toUpperCase()) {
            return correctedWord.charAt(0).toUpperCase() + correctedWord.slice(1);
        }

        // Default to lowercase
        return correctedWord;
    }

    // --- Helper Function: Move Cursor ---
    function moveCursorToEnd(element) {
        if (element.value !== undefined) {
            // For textarea/input
            element.selectionStart = element.selectionEnd = element.value.length;
            element.focus();
        } else if (element.isContentEditable) {
            // For contentEditable elements
            const range = document.createRange();
            const selection = window.getSelection();
            range.selectNodeContents(element);
            range.collapse(false); 
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }

    // --- Sandbox Communication Setup ---

    function setupSandbox() {
        sandboxIframe = document.createElement('iframe');
        sandboxIframe.src = chrome.runtime.getURL('sandbox.html');
        sandboxIframe.style.display = 'none'; // Keep it invisible
        sandboxIframe.setAttribute('sandbox', 'allow-scripts'); // Explicit sandbox attribute
        document.body.appendChild(sandboxIframe);

        // Listener to receive messages from the sandbox
        window.addEventListener('message', handleSandboxMessage);
        
        console.log("Content Script: Sandbox iframe created and listener attached");
    }

    function handleSandboxMessage(event) {
        // Security check: Only accept messages from the sandboxed iframe
        if (!event.data || event.source !== sandboxIframe.contentWindow) {
            return; 
        }

        const messageType = event.data.type;

        if (messageType === 'GTC_SANDBOX_LOADED') {
            // Sandbox iframe is loaded, now send initialization data
            console.log("Content Script: Sandbox loaded, sending initialization data...");
            // The sandbox loads the model (weights.json, tokenizer, dictionary)
            // from this directory via plain-JS inference.
            const modelBase = chrome.runtime.getURL('model/');

            sandboxIframe.contentWindow.postMessage({
                type: 'GTC_INIT',
                modelBase: modelBase
            }, '*');
        } else if (messageType === 'GTC_READY') {
            // AI model is loaded and ready to receive prediction requests
            sandboxReady = true;
            console.log("Content Script: Sandbox is ready! AI model loaded");
            // Attach event listeners once everything is ready
            attachListeners(); 
        } else if (messageType === 'GTC_ERROR') {
            console.error("Content Script: Sandbox Error:", event.data.message);
        } else if (messageType === 'GTC_RESULT') {
            // Correction result received from the sandbox
            applyCorrection(event.data.originalWord, event.data.correctedWord);
        }
    }

    // --- Autocorrect Logic ---

    function attachListeners() {
        document.body.addEventListener('keyup', handleKeyUp, true);
        document.body.addEventListener('keydown', handleKeyDownForUndo, true);
        console.log("Content Script: Event listeners attached");
    }

    function handleKeyUp(event) {
        // Trigger only on spacebar AND if sandbox is ready
        if (event.key !== ' ' || !sandboxReady) {
            return;
        }

        const activeElement = event.target;
        
        // Check if the active element is a text input field
        if (activeElement.tagName.toLowerCase() !== 'textarea' && 
            activeElement.type !== 'text' && 
            activeElement.type !== 'search' && 
            activeElement.type !== 'email' &&
            activeElement.type !== 'url' &&
            !activeElement.isContentEditable) {
            return;
        }

        const text = activeElement.value || activeElement.textContent;
        if (!text) return;

        const words = text.trim().split(/\s+/);
        if (words.length === 0) return;

        const wordToCheck = words[words.length - 1];
        if (!wordToCheck || wordToCheck.length === 0) return;

        lastProcessedElement = activeElement; // Store the element for later correction/undo

        // Send word to the sandbox for prediction
        sandboxIframe.contentWindow.postMessage({
            type: 'GTC_PREDICT',
            word: wordToCheck
        }, '*');
    }

    // --- Apply Correction from Sandbox Result ---

    function applyCorrection(originalWord, rawCorrection) {
        if (!rawCorrection || 
            !originalWord || 
            !lastProcessedElement ||
            originalWord.toLowerCase() === rawCorrection.toLowerCase()) {
            return;
        }
        
        const activeElement = lastProcessedElement;
        const finalCorrection = matchCase(originalWord, rawCorrection);
        
        const text = activeElement.value || activeElement.textContent;
        if (!text) return;

        const words = text.trim().split(/\s+/);
        
        // Ensure the word hasn't changed since we sent the request
        if (words.length === 0 || words[words.length - 1].toLowerCase() !== originalWord.toLowerCase()) {
            // User kept typing, ignore stale correction
            return;
        }
        
        // Perform replacement
        words[words.length - 1] = finalCorrection;
        const newText = words.join(' ') + ' ';

        if (activeElement.value !== undefined) {
            activeElement.value = newText;
        } else {
            activeElement.textContent = newText;
            moveCursorToEnd(activeElement);
        }

        // Track autocorrect for undo feature
        lastOriginalWord = originalWord;
        lastCorrectedWord = finalCorrection;

        console.log(`Ghost Type Corrector: "${originalWord}" → "${finalCorrection}"`);
    }

    // --- Undo Autocorrect Feature ---
    function handleKeyDownForUndo(event) {
        if (event.key === 'Backspace' && lastCorrectedWord && lastProcessedElement) {
            const activeElement = document.activeElement;
            
            // Check if the user is in the corrected element
            if (activeElement === lastProcessedElement) {
                const text = activeElement.value || activeElement.textContent;
                if (!text) return;

                const words = text.trim().split(/\s+/);
                
                // Check if the last word matches the corrected word
                if (words.length > 0 && words[words.length - 1] === lastCorrectedWord) {
                    // Undo autocorrect
                    words[words.length - 1] = lastOriginalWord;
                    const newText = words.join(' ') + ' ';
                    
                    if (activeElement.value !== undefined) {
                        activeElement.value = newText;
                    } else {
                        activeElement.textContent = newText;
                        moveCursorToEnd(activeElement);
                    }
                    
                    console.log(`Ghost Type Corrector: Undo - "${lastCorrectedWord}" → "${lastOriginalWord}"`);

                    // Clear last correction data so it only works once
                    lastOriginalWord = null;
                    lastCorrectedWord = null;
                    lastProcessedElement = null;

                    // Prevent default backspace action
                    event.preventDefault();
                }
            }
        }
    }
    
    // --- Initialization Entry Point ---
    function initialize() {
        console.log("Ghost Type Corrector: Initializing...");
        
        // Setup sandbox for AI model
        setupSandbox();
        
        console.log("Ghost Type Corrector: Waiting for Sandbox AI model to load...");
    }

    // Start the extension setup when DOM is ready
    if (document.body) {
        initialize();
    } else {
        // Wait for body to be available
        const observer = new MutationObserver((mutations, obs) => {
            if (document.body) {
                initialize();
                obs.disconnect();
            }
        });
        observer.observe(document.documentElement, { childList: true });
    }

})();