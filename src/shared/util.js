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
import * as tf from '@tensorflow/tfjs-core';
import {showBackendConfigs} from './option_panel';
import {GREEN, NUM_KEYPOINTS, RED, STATE, TUNABLE_FLAG_VALUE_RANGE_MAP} from './params';

export function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

export function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

export function isMobile() {
  return isAndroid() || isiOS();
}

/**
 * Reset the target backend.
 *
 * @param backendName The name of the backend to be reset.
 */
async function resetBackend(backendName) {
  const ENGINE = tf.engine();
  if (!(backendName in ENGINE.registryFactory)) {
    if(backendName === 'webgpu') {
      alert('webgpu backend is not registered. Your browser may not support WebGPU yet. To test this backend, please use a supported browser, e.g. Chrome canary with --enable-unsafe-webgpu flag');
      STATE.backend = !!STATE.lastTFJSBackend ? STATE.lastTFJSBackend : 'tfjs-webgl';
      showBackendConfigs();
      return;
    } else {
      throw new Error(`${backendName} backend is not registered.`);
    }
  }

  if (backendName in ENGINE.registry) {
    const backendFactory = tf.findBackendFactory(backendName);
    tf.removeBackend(backendName);
    tf.registerBackend(backendName, backendFactory);
  }

  await tf.setBackend(backendName);
  STATE.lastTFJSBackend = `tfjs-${backendName}`;
}

/**
 * Set environment flags.
 *
 * This is a wrapper function of `tf.env().setFlags()` to constrain users to
 * only set tunable flags (the keys of `TUNABLE_FLAG_TYPE_MAP`).
 *
 * ```js
 * const flagConfig = {
 *        WEBGL_PACK: false,
 *      };
 * await setEnvFlags(flagConfig);
 *
 * console.log(tf.env().getBool('WEBGL_PACK')); // false
 * console.log(tf.env().getBool('WEBGL_PACK_BINARY_OPERATIONS')); // false
 * ```
 *
 * @param flagConfig An object to store flag-value pairs.
 */
export async function setBackendAndEnvFlags(flagConfig, backend) {
  if (flagConfig == null) {
    return;
  } else if (typeof flagConfig !== 'object') {
    throw new Error(
        `An object is expected, while a(n) ${typeof flagConfig} is found.`);
  }

  // Check the validation of flags and values.
  for (const flag in flagConfig) {
    // TODO: check whether flag can be set as flagConfig[flag].
    if (!(flag in TUNABLE_FLAG_VALUE_RANGE_MAP)) {
      throw new Error(`${flag} is not a tunable or valid environment flag.`);
    }
    if (TUNABLE_FLAG_VALUE_RANGE_MAP[flag].indexOf(flagConfig[flag]) === -1) {
      throw new Error(
          `${flag} value is expected to be in the range [${
              TUNABLE_FLAG_VALUE_RANGE_MAP[flag]}], while ${flagConfig[flag]}` +
          ' is found.');
    }
  }

  tf.env().setFlags(flagConfig);

  const [runtime, $backend] = backend.split('-');

  if (runtime === 'tfjs') {
    await resetBackend($backend);
  }
}

function distance(a, b) {
  return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

function drawPath(ctx, points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

/**
 * Draw the keypoints on the video.
 * @param ctx 2D rendering context.
 * @param faces A list of faces to render.
 * @param boundingBox Whether or not to display the bounding box.
 * @param showKeypoints Whether or not to display the keypoints.
 */
/**
 * Check if a face is centered in the viewport
 * @param {Object} face The detected face object
 * @param {number} canvasWidth The width of the canvas
 * @param {number} canvasHeight The height of the canvas
 * @returns {boolean} Whether the face is centered
 */
function isFaceCentered(face, canvasWidth, canvasHeight) {
  const box = face.box;
  const faceWidth = box.xMax - box.xMin;
  const faceHeight = box.yMax - box.yMin;
  
  // Calculate the center of the face
  const faceCenterX = box.xMin + faceWidth / 2;
  const faceCenterY = box.yMin + faceHeight / 2;
  
  // Calculate the center of the canvas
  const canvasCenterX = canvasWidth / 2;
  const canvasCenterY = canvasHeight / 2;
  
  // Calculate the distance from the face center to the canvas center
  const distanceX = Math.abs(faceCenterX - canvasCenterX);
  const distanceY = Math.abs(faceCenterY - canvasCenterY);
  
  // Define thresholds for centering (adjust as needed)
  const thresholdX = canvasWidth * 0.15; // 15% of canvas width
  const thresholdY = canvasHeight * 0.15; // 15% of canvas height
  
  // Check if the face is centered within the thresholds
  return distanceX <= thresholdX && distanceY <= thresholdY;
}

// Keep track of the last announced status to avoid repetitive announcements
let lastAnnouncedStatus = '';
let lastAnnouncementTime = 0;

/**
 * Speak a message using the Web Speech API
 * @param {string} message The message to speak
 * @param {boolean} interrupt Whether to interrupt ongoing speech
 */
function speakMessage(message, interrupt = false) {
  // Check if speech synthesis is available
  if (!('speechSynthesis' in window)) {
    console.warn('Speech synthesis not supported in this browser');
    return;
  }
  
  // Don't repeat the same message too frequently (at least 3 seconds between identical announcements)
  const now = Date.now();
  if (message === lastAnnouncedStatus && now - lastAnnouncementTime < 3000) {
    return;
  }
  
  // If interrupt is true, cancel any ongoing speech
  if (interrupt && window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
  }
  
  // Create a new speech utterance
  const utterance = new SpeechSynthesisUtterance(message);
  
  // Set properties for better accessibility
  utterance.volume = 1; // 0 to 1
  utterance.rate = 1.1; // 0.1 to 10
  utterance.pitch = 1; // 0 to 2
  utterance.lang = 'en-US';
  
  // Add event listeners to debug speech synthesis issues
  utterance.onstart = () => console.log('Speech started: ' + message);
  utterance.onend = () => console.log('Speech ended: ' + message);
  utterance.onerror = (event) => console.error('Speech error:', event);
  
  // Force speech synthesis to start - sometimes browsers require user interaction first
  try {
    // Ensure speech synthesis is ready
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
    }
    
    // Speak the message
    window.speechSynthesis.speak(utterance);
    
    // Log for debugging
    console.log('Speaking message: ' + message);
    
    // Update tracking variables
    lastAnnouncedStatus = message;
    lastAnnouncementTime = now;
  } catch (error) {
    console.error('Error speaking message:', error);
  }
}

/**
 * Update the face centering status in the UI and announce it via speech
 * @param {boolean} isCentered Whether the face is centered
 */
function updateFaceStatus(isCentered) {
  const statusElement = document.getElementById('face-status');
  if (!statusElement) return;
  
  let statusMessage = '';
  
  if (isCentered) {
    statusMessage = 'Face position: Centered';
    statusElement.textContent = statusMessage + ' ✅';
    statusElement.style.color = 'green';
    speakMessage('Your face is centered. You can take a photo now.', true);
    
    // Focus on the 'take-and-download' button when face is centered
    const takeAndDownloadButton = document.getElementById('take-and-download');
    if (takeAndDownloadButton) {
      takeAndDownloadButton.focus();
    }
  } else {
    statusMessage = 'Face position: Not centered - Please center your face';
    statusElement.textContent = statusMessage + ' ❌';
    statusElement.style.color = 'red';
    
    // Use the guidance that was calculated in drawResults
    if (window.currentFaceGuidance) {
      speakMessage(window.currentFaceGuidance);
    } else {
      speakMessage('Your face is not centered. Please adjust your position.');
    }
  }
}

/**
 * Get more specific guidance on how to adjust face position
 * @param {Object} face The detected face object
 * @param {number} canvasWidth The width of the canvas
 * @param {number} canvasHeight The height of the canvas
 * @returns {string} Guidance message with directional instructions
 */
function getFacePositionGuidance(face, canvasWidth, canvasHeight) {
  if (!face || !face.box) return '';
  
  const box = face.box;
  const faceWidth = box.xMax - box.xMin;
  const faceHeight = box.yMax - box.yMin;
  
  // Calculate the center of the face
  const faceCenterX = box.xMin + faceWidth / 2;
  const faceCenterY = box.yMin + faceHeight / 2;
  
  // Calculate the center of the canvas
  const canvasCenterX = canvasWidth / 2;
  const canvasCenterY = canvasHeight / 2;
  
  // Calculate the distance from the face center to the canvas center
  const distanceX = faceCenterX - canvasCenterX;
  const distanceY = faceCenterY - canvasCenterY;
  
  // Determine horizontal position
  let horizontalGuidance = '';
  if (distanceX < -canvasWidth * 0.1) {
    horizontalGuidance = 'Move right';
  } else if (distanceX > canvasWidth * 0.1) {
    horizontalGuidance = 'Move left';
  }
  
  // Determine vertical position
  let verticalGuidance = '';
  if (distanceY < -canvasHeight * 0.1) {
    verticalGuidance = 'Move down';
  } else if (distanceY > canvasHeight * 0.1) {
    verticalGuidance = 'Move up';
  }
  
  // Determine distance guidance (if face is too close or too far)
  let distanceGuidance = '';
  const idealFaceWidth = canvasWidth * 0.4; // Face should take up about 40% of frame width
  if (faceWidth < idealFaceWidth * 0.7) {
    distanceGuidance = 'Move closer to the camera';
  } else if (faceWidth > idealFaceWidth * 1.3) {
    distanceGuidance = 'Move farther from the camera';
  }
  
  // Combine the guidance messages
  const guidanceParts = [horizontalGuidance, verticalGuidance, distanceGuidance].filter(part => part !== '');
  
  if (guidanceParts.length === 0) {
    return 'Almost centered. Make small adjustments.';
  }
  
  return guidanceParts.join('. ') + '.';
}

export function drawResults(ctx, faces, boundingBox, showKeypoints) {
  // Get canvas dimensions
  const canvasWidth = ctx.canvas.width;
  const canvasHeight = ctx.canvas.height;
  
  if (faces.length === 0) {
    // No face detected
    const statusElement = document.getElementById('face-status');
    if (statusElement) {
      statusElement.textContent = 'Face position: No face detected';
      statusElement.style.color = 'orange';
    }
  }
  
  faces.forEach((face) => {
    const keypoints =
        face.keypoints.map((keypoint) => [keypoint.x, keypoint.y]);

    if (boundingBox) {
      ctx.strokeStyle = RED;
      ctx.lineWidth = 1;

      const box = face.box;
      drawPath(
          ctx,
          [
            [box.xMin, box.yMin], [box.xMax, box.yMin], [box.xMax, box.yMax],
            [box.xMin, box.yMax]
          ],
          true);
      
      // Check if the face is centered and update the status
      const centered = isFaceCentered(face, canvasWidth, canvasHeight);
      
      // Get directional guidance for this face
      window.currentFaceGuidance = getFacePositionGuidance(face, canvasWidth, canvasHeight);
      
      // Update the status with centering information
      updateFaceStatus(centered);
    }

    if (showKeypoints) {
      ctx.fillStyle = GREEN;

      for (let i = 0; i < NUM_KEYPOINTS; i++) {
        const x = keypoints[i][0];
        const y = keypoints[i][1];

        ctx.beginPath();
        ctx.arc(x, y, 3 /* radius */, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  });
}