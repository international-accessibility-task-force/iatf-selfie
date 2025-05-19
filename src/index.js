/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as faceDetection from '@tensorflow-models/face-detection';

import {Camera} from './camera';
import {setupDatGui} from './option_panel';
import {STATE, createDetector} from './shared/params';
import {setupStats} from './shared/stats_panel';
import {setBackendAndEnvFlags} from './shared/util';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimateFaceStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimateFaceStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let faces = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimateFaces.
    beginEstimateFaceStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      faces =
          await detector.estimateFaces(camera.video, {flipHorizontal: false});
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    endEstimateFaceStats();
  }

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (faces && faces.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(
        faces, STATE.modelConfig.boundingBox, STATE.modelConfig.keypoints);
  }
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(renderPrediction);
};

// Function to capture a photo when the face is centered
function capturePhoto() {
  const statusElement = document.getElementById('face-status');
  const takePhotoButton = document.getElementById('take-photo');
  const downloadPhotoButton = document.getElementById('download-photo');
  const photoCanvas = document.getElementById('photo-canvas');
  const capturedPhoto = document.getElementById('captured-photo');
  
  // Check if face is centered
  if (statusElement && statusElement.textContent.includes('Centered')) {
    // Set the photo canvas dimensions to match the video
    photoCanvas.width = camera.video.videoWidth;
    photoCanvas.height = camera.video.videoHeight;
    
    // Get the canvas context and draw the current video frame
    const ctx = photoCanvas.getContext('2d');
    
    // Since the video is mirrored, we need to mirror the canvas too
    ctx.translate(photoCanvas.width, 0);
    ctx.scale(-1, 1);
    
    // Draw the video frame to the canvas
    ctx.drawImage(camera.video, 0, 0, photoCanvas.width, photoCanvas.height);
    
    // Convert canvas to image URL
    const imageUrl = photoCanvas.toDataURL('image/png');
    
    // Display the captured image
    capturedPhoto.src = imageUrl;
    capturedPhoto.style.display = 'block';
    
    // Enable download button
    downloadPhotoButton.disabled = false;
    
    // Play a camera shutter sound for audio feedback
    playShutterSound();
    
    // Announce success for screen reader users
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel(); // Cancel any ongoing speech
      const successMessage = 'Photo captured successfully. You can now download it using the Download Photo button.';
      const utterance = new SpeechSynthesisUtterance(successMessage);
      utterance.rate = 1.1;
      utterance.volume = 1;
      window.speechSynthesis.speak(utterance);
    } else {
      // Fallback to alert if speech synthesis is not available
      alert('Photo captured successfully!');
    }
  } else {
    // Provide audio feedback if face is not centered
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel(); // Cancel any ongoing speech
      const warningMessage = 'Please center your face before taking a photo. Use the Announce Position button for guidance.';
      const utterance = new SpeechSynthesisUtterance(warningMessage);
      utterance.rate = 1.1;
      utterance.volume = 1;
      window.speechSynthesis.speak(utterance);
    } else {
      // Fallback to alert if speech synthesis is not available
      alert('Please center your face before taking a photo.');
    }
  }
}

// Function to set up UI elements based on environment (dev vs prod)
function setupUIForEnvironment(isDev) {
  const takePhotoButton = document.getElementById('take-photo');
  const downloadPhotoButton = document.getElementById('download-photo');
  const announcePositionButton = document.getElementById('announce-position');
  const takeAndDownloadButton = document.getElementById('take-and-download');
  const faceStatusElement = document.getElementById('face-status');
  
  if (isDev) {
    // In development mode - show all controls
    if (takePhotoButton) takePhotoButton.style.display = 'inline-block';
    if (downloadPhotoButton) downloadPhotoButton.style.display = 'inline-block';
    if (announcePositionButton) announcePositionButton.style.display = 'inline-block';
    if (takeAndDownloadButton) takeAndDownloadButton.style.display = 'inline-block';
    
    // Show development elements
    const statsElement = document.getElementById('stats');
    if (statsElement) statsElement.style.display = 'block';
    
    // Make sure dat.gui is visible
    const datGuiElement = document.querySelector('.dg.ac');
    if (datGuiElement) datGuiElement.style.display = 'block';
    
    // Add Test Voice button in dev mode if it doesn't exist
    if (!document.getElementById('test-voice')) {
      const voiceTestButton = document.createElement('button');
      voiceTestButton.id = 'test-voice';
      voiceTestButton.style.cssText = 'padding: 10px 20px; font-size: 16px; background-color: #673AB7; color: white; border: none; border-radius: 5px; cursor: pointer; margin-right: 10px;';
      voiceTestButton.setAttribute('aria-label', 'Test voice synthesis');
      voiceTestButton.textContent = 'Test Voice';
      voiceTestButton.addEventListener('click', () => {
        if ('speechSynthesis' in window) {
          const testMessage = 'Voice synthesis test. This is how the app will speak to you.';
          const utterance = new SpeechSynthesisUtterance(testMessage);
          utterance.rate = 1.1;
          utterance.pitch = 1;
          utterance.volume = 1;
          window.speechSynthesis.cancel(); // Cancel any ongoing speech
          window.speechSynthesis.speak(utterance);
        }
      });
      
      // Add before the Take Photo button
      if (takePhotoButton && takePhotoButton.parentNode) {
        takePhotoButton.parentNode.insertBefore(voiceTestButton, takePhotoButton);
      }
    }
  } else {
    // In production mode - only show Take & Download Photo button
    if (takePhotoButton) takePhotoButton.style.display = 'none';
    if (downloadPhotoButton) downloadPhotoButton.style.display = 'none';
    if (announcePositionButton) announcePositionButton.style.display = 'none';
    
    // Hide development elements
    const statsElement = document.getElementById('stats');
    if (statsElement) statsElement.style.display = 'none';
    
    // Hide dat.gui interface
    const datGuiElement = document.querySelector('.dg.ac');
    if (datGuiElement) datGuiElement.style.display = 'none';
    
    // Make sure Take & Download button is visible and styled prominently
    if (takeAndDownloadButton) {
      takeAndDownloadButton.style.display = 'block';
      takeAndDownloadButton.style.margin = '10px auto';
      takeAndDownloadButton.style.fontSize = '18px';
      takeAndDownloadButton.style.padding = '15px 25px';
    }
    
    // Remove any Test Voice button if it exists
    const testVoiceButton = document.getElementById('test-voice');
    if (testVoiceButton && testVoiceButton.parentNode) {
      testVoiceButton.parentNode.removeChild(testVoiceButton);
    }
  }
}

// Function to play a camera shutter sound for audio feedback
function playShutterSound() {
  // Create an audio context
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) return; // Browser doesn't support Web Audio API
  
  const audioContext = new AudioContext();
  
  // Create an oscillator for a quick 'click' sound
  const oscillator = audioContext.createOscillator();
  const gainNode = audioContext.createGain();
  
  oscillator.type = 'sine';
  oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
  oscillator.frequency.exponentialRampToValueAtTime(300, audioContext.currentTime + 0.1);
  
  gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
  
  oscillator.connect(gainNode);
  gainNode.connect(audioContext.destination);
  
  oscillator.start();
  oscillator.stop(audioContext.currentTime + 0.1);
}

// Function to download the captured photo
function downloadPhoto() {
  const photoCanvas = document.getElementById('photo-canvas');
  
  if (photoCanvas) {
    const filename = 'selfie_' + new Date().toISOString().replace(/[:.]/g, '-') + '.png';
    const imageUrl = photoCanvas.toDataURL('image/png');
    
    // Check if it's a mobile device
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    if (isMobile && (navigator.canShare || navigator.share)) {
      // Convert the data URL to a blob
      fetch(imageUrl)
        .then(res => res.blob())
        .then(blob => {
          // Create a File object
          const file = new File([blob], filename, { type: 'image/png' });
          
          // Use Web Share API to save to gallery
          if (navigator.canShare && navigator.canShare({ files: [file] })) {
            navigator.share({
              files: [file],
              title: 'Selfie',
              text: 'My selfie taken with the Accessible Selfie App'
            })
            .then(() => console.log('Shared successfully'))
            .catch((error) => {
              console.error('Error sharing:', error);
              // Fall back to regular download if sharing fails
              fallbackDownload(imageUrl, filename);
            });
          } else if (navigator.share) {
            // Fallback for browsers that support share but not file sharing
            navigator.share({
              title: 'Selfie',
              text: 'My selfie taken with the Accessible Selfie App',
              url: imageUrl
            })
            .then(() => console.log('Shared successfully'))
            .catch((error) => {
              console.error('Error sharing:', error);
              fallbackDownload(imageUrl, filename);
            });
          } else {
            fallbackDownload(imageUrl, filename);
          }
        });
    } else {
      // For desktop or browsers without sharing capability
      fallbackDownload(imageUrl, filename);
    }
  }
}

// Fallback download function for non-mobile devices or when sharing fails
function fallbackDownload(imageUrl, filename) {
  const link = document.createElement('a');
  link.download = filename;
  link.href = imageUrl;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Function to take and download a photo in a single action
function takeAndDownloadPhoto() {
  const statusElement = document.getElementById('face-status');
  const photoCanvas = document.getElementById('photo-canvas');
  const capturedPhoto = document.getElementById('captured-photo');
  
  // Check if face is centered
  if (statusElement && statusElement.textContent.includes('Centered')) {
    // Set the photo canvas dimensions to match the video
    photoCanvas.width = camera.video.videoWidth;
    photoCanvas.height = camera.video.videoHeight;
    
    // Get the canvas context and draw the current video frame
    const ctx = photoCanvas.getContext('2d');
    
    // Since the video is mirrored, we need to mirror the canvas too
    ctx.translate(photoCanvas.width, 0);
    ctx.scale(-1, 1);
    
    // Draw the video frame to the canvas
    ctx.drawImage(camera.video, 0, 0, photoCanvas.width, photoCanvas.height);
    
    // Convert canvas to image URL
    const imageUrl = photoCanvas.toDataURL('image/png');
    
    // Display the captured image
    capturedPhoto.src = imageUrl;
    capturedPhoto.style.display = 'block';
    
    // Enable download button (even though we're downloading automatically)
    const downloadPhotoButton = document.getElementById('download-photo');
    if (downloadPhotoButton) {
      downloadPhotoButton.disabled = false;
    }
    
    // Play a camera shutter sound for audio feedback
    playShutterSound();
    
    // Save the photo
    const filename = 'selfie_' + new Date().toISOString().replace(/[:.]/g, '-') + '.png';
    
    // Check if it's a mobile device
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    // Announce success for screen reader users
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel(); // Cancel any ongoing speech
      const successMessage = isMobile ? 'Photo captured. Saving to gallery.' : 'Photo captured and downloading automatically.';
      const utterance = new SpeechSynthesisUtterance(successMessage);
      utterance.rate = 1.1;
      utterance.volume = 1;
      window.speechSynthesis.speak(utterance);
      
      // Save the photo after a short delay to allow the speech to be heard
      setTimeout(() => {
        savePhoto(imageUrl, filename);
        
        // Disable the app and show start over button after photo is taken and downloaded
        disableAppAndShowStartOver();
      }, 1500);
    } else {
      // If speech synthesis is not available, save immediately
      savePhoto(imageUrl, filename);
      alert(isMobile ? 'Photo captured. Saving to gallery.' : 'Photo captured and downloaded successfully!');
      
      // Disable the app and show start over button after photo is taken and downloaded
      disableAppAndShowStartOver();
    }
    
    function savePhoto(imageUrl, filename) {
      if (isMobile && (navigator.canShare || navigator.share)) {
        // Convert the data URL to a blob
        fetch(imageUrl)
          .then(res => res.blob())
          .then(blob => {
            // Create a File object
            const file = new File([blob], filename, { type: 'image/png' });
            
            // Use Web Share API to save to gallery
            if (navigator.canShare && navigator.canShare({ files: [file] })) {
              navigator.share({
                files: [file],
                title: 'Selfie',
                text: 'My selfie taken with the Accessible Selfie App'
              })
              .then(() => console.log('Shared successfully'))
              .catch((error) => {
                console.error('Error sharing:', error);
                // Fall back to regular download if sharing fails
                fallbackDownload(imageUrl, filename);
              });
            } else if (navigator.share) {
              // Fallback for browsers that support share but not file sharing
              navigator.share({
                title: 'Selfie',
                text: 'My selfie taken with the Accessible Selfie App',
                url: imageUrl
              })
              .then(() => console.log('Shared successfully'))
              .catch((error) => {
                console.error('Error sharing:', error);
                fallbackDownload(imageUrl, filename);
              });
            } else {
              fallbackDownload(imageUrl, filename);
            }
          });
      } else {
        // For desktop or browsers without sharing capability
        fallbackDownload(imageUrl, filename);
      }
    }
  } else {
    // Provide audio feedback if face is not centered
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel(); // Cancel any ongoing speech
      const warningMessage = 'Please center your face before taking a photo. Use the Announce Position button for guidance.';
      const utterance = new SpeechSynthesisUtterance(warningMessage);
      utterance.rate = 1.1;
      utterance.volume = 1;
      window.speechSynthesis.speak(utterance);
    } else {
      // Fallback to alert if speech synthesis is not available
      alert('Please center your face before taking a photo.');
    }
  }
}

// Function to disable the app and show Start Over button
function disableAppAndShowStartOver() {
  // Cancel the animation frame to stop the camera processing
  window.cancelAnimationFrame(rafId);
  
  // Hide the Take & Download Photo button
  const takeAndDownloadButton = document.getElementById('take-and-download');
  if (takeAndDownloadButton) {
    takeAndDownloadButton.style.display = 'none';
  }
  
  // Check if Start Over button already exists, create it if not
  let startOverButton = document.getElementById('start-over');
  if (!startOverButton) {
    startOverButton = document.createElement('button');
    startOverButton.id = 'start-over';
    startOverButton.textContent = 'Take Another Selfie';
    startOverButton.setAttribute('aria-label', 'Take another selfie');
    startOverButton.style.padding = '10px 20px';
    startOverButton.style.fontSize = '16px';
    startOverButton.style.backgroundColor = '#673AB7';
    startOverButton.style.color = 'white';
    startOverButton.style.border = 'none';
    startOverButton.style.borderRadius = '5px';
    startOverButton.style.cursor = 'pointer';
    startOverButton.style.marginTop = '10px';
    startOverButton.style.width = '100%';
    startOverButton.style.maxWidth = '300px';
    
    // Add the button to the page
    const buttonContainer = document.querySelector('.container div[style*="text-align: center"]');
    if (buttonContainer) {
      buttonContainer.appendChild(startOverButton);
    }
    
    // Add event listener
    startOverButton.addEventListener('click', startOver);
  } else {
    // Show the existing Start Over button
    startOverButton.style.display = 'block';
    startOverButton.textContent = 'Take Another Selfie';
  }
  
  // Set focus on the button
  setTimeout(() => {
    if (startOverButton) {
      startOverButton.focus();
    }
  }, 100);
  
  // Announce for screen readers
  if ('speechSynthesis' in window) {
    const message = 'Photo has been taken and downloaded. Press the Take Another Selfie button to take a new photo.';
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 1.1;
    utterance.volume = 1;
    window.speechSynthesis.speak(utterance);
  }
}

// Function to start over and reset the app
function startOver() {
  // Hide the captured photo
  const capturedPhoto = document.getElementById('captured-photo');
  if (capturedPhoto) {
    capturedPhoto.style.display = 'none';
  }
  
  // Show the Take & Download Photo button again
  const takeAndDownloadButton = document.getElementById('take-and-download');
  if (takeAndDownloadButton) {
    takeAndDownloadButton.style.display = 'block';
  }
  
  // Hide the Start Over button
  const startOverButton = document.getElementById('start-over');
  if (startOverButton) {
    startOverButton.style.display = 'none';
  }
  
  // Restart the camera processing
  rafId = requestAnimationFrame(renderPrediction);
  
  // Announce for screen readers
  if ('speechSynthesis' in window) {
    const message = 'App restarted. You can now take a new photo.';
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 1.1;
    utterance.volume = 1;
    window.speechSynthesis.speak(utterance);
  }
}

// Initialize speech synthesis to ensure it works
function initSpeechSynthesis() {
  if ('speechSynthesis' in window) {
    // Some browsers require a user interaction before allowing speech synthesis
    // This function creates a silent utterance to initialize the speech system
    const silentUtterance = new SpeechSynthesisUtterance('');
    silentUtterance.volume = 0;
    window.speechSynthesis.speak(silentUtterance);
    
    // Create a test voice button for troubleshooting
    const container = document.querySelector('.container');
    if (container) {
      const testVoiceButton = document.createElement('button');
      testVoiceButton.id = 'test-voice';
      testVoiceButton.textContent = 'Test Voice';
      testVoiceButton.setAttribute('aria-label', 'Test voice output');
      testVoiceButton.style.padding = '10px 20px';
      testVoiceButton.style.fontSize = '16px';
      testVoiceButton.style.backgroundColor = '#673AB7';
      testVoiceButton.style.color = 'white';
      testVoiceButton.style.border = 'none';
      testVoiceButton.style.borderRadius = '5px';
      testVoiceButton.style.cursor = 'pointer';
      testVoiceButton.style.marginTop = '10px';
      
      // Add event listener to test voice button
      testVoiceButton.addEventListener('click', () => {
        const testMessage = 'Voice output is working. You can now use the face detection app with voice guidance.';
        const utterance = new SpeechSynthesisUtterance(testMessage);
        utterance.volume = 1;
        utterance.rate = 1.1;
        utterance.pitch = 1;
        utterance.lang = 'en-US';
        
        // Cancel any ongoing speech
        window.speechSynthesis.cancel();
        
        // Speak the test message
        window.speechSynthesis.speak(utterance);
      });
      
      // Add the button to the page
      const buttonContainer = document.createElement('div');
      buttonContainer.style.textAlign = 'center';
      buttonContainer.style.marginTop = '10px';
      buttonContainer.appendChild(testVoiceButton);
      container.appendChild(buttonContainer);
    }
  }
}

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);

  // Determine if we're in development or production mode
  const isDev = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  
  // Setup UI based on environment
  setupUIForEnvironment(isDev);

  // Only setup dat.gui in dev mode
  if (isDev) {
    await setupDatGui(urlParams);
    stats = setupStats();
  } else {
    // Hide stats in production
    const statsElement = document.getElementById('stats');
    if (statsElement) statsElement.style.display = 'none';
  }

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();
  
  // Initialize speech synthesis
  initSpeechSynthesis();
  
  // Add event listeners for the photo capture and download buttons
  const takePhotoButton = document.getElementById('take-photo');
  const downloadPhotoButton = document.getElementById('download-photo');
  const announcePositionButton = document.getElementById('announce-position');
  const takeAndDownloadButton = document.getElementById('take-and-download');
  const startOverButton = document.getElementById('start-over');
  
  if (takePhotoButton) {
    takePhotoButton.addEventListener('click', capturePhoto);
  }
  
  if (downloadPhotoButton) {
    downloadPhotoButton.addEventListener('click', downloadPhoto);
  }
  
  // Add event listener for the combined take and download button
  if (takeAndDownloadButton) {
    takeAndDownloadButton.addEventListener('click', takeAndDownloadPhoto);
  }
  
  // Add event listener for the start over button if it exists
  if (startOverButton) {
    startOverButton.addEventListener('click', startOver);
  }
  
  // Add event listener for the announce position button
  if (announcePositionButton) {
    announcePositionButton.addEventListener('click', () => {
      // Force announcement of current face position
      const statusElement = document.getElementById('face-status');
      if (statusElement) {
        // Extract the text content without the emoji
        const statusText = statusElement.textContent.replace('✅', '').replace('❌', '').trim();
        
        // Create a more detailed message for blind users
        let message = statusText;
        
        // Add the directional guidance if available
        if (window.currentFaceGuidance) {
          message += '. ' + window.currentFaceGuidance;
        }
        
        // Add instructions on how to take a photo
        if (statusText.includes('Centered')) {
          message += '. You can now take a photo by pressing the Take Photo button in the middle of the screen.';
        }
        
        // Use the speech synthesis API to announce the position
        if ('speechSynthesis' in window) {
          const utterance = new SpeechSynthesisUtterance(message);
          utterance.rate = 1.1;
          utterance.pitch = 1;
          utterance.volume = 1;
          window.speechSynthesis.cancel(); // Cancel any ongoing speech
          window.speechSynthesis.speak(utterance);
        }
      }
    });
  }

  renderPrediction();
};

app();
