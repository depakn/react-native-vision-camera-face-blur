package com.mrousavy.camera.core

import android.annotation.SuppressLint
import android.util.Log
import android.util.Size
import androidx.annotation.OptIn
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis.Analyzer
import androidx.camera.core.ImageProxy
import androidx.camera.video.Recording
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.util.concurrent.Executor
import java.util.concurrent.Executors

class FrameProcessorPipeline(
  private var isFrontFacing: Boolean,
  private val videoProcessor: VideoProcessor?,
  private val videoEncoder: VideoEncoder?
) : Analyzer {
  private val processingExecutor: Executor = Executors.newSingleThreadExecutor()
  private var lastProcessingTimestamp: Long = 0

  private val options = FaceDetectorOptions.Builder()
    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
    .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
    .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
    .setMinFaceSize(0.15f)
    .build()
  private val detector = FaceDetection.getClient(options)

  fun updateIsFrontFacing(newIsFrontFacing: Boolean) {
    isFrontFacing = newIsFrontFacing
  }

  @OptIn(ExperimentalGetImage::class)
  override fun analyze(imageProxy: ImageProxy) {
    val currentTimestamp = System.currentTimeMillis()
    if (currentTimestamp - lastProcessingTimestamp < MINIMUM_PROCESSING_INTERVAL_MS) {
      imageProxy.close()
      return
    }

    processingExecutor.execute {
      try {
        detectFaces(imageProxy)
      } catch (e: Exception) {
        Log.e(TAG, "Error processing frame: ${e.message}")
        imageProxy.close()
      }
    }

    lastProcessingTimestamp = currentTimestamp
  }

  private fun processDetectedFaces(faces: List<Face>, imageProxy: ImageProxy) {
    try {
      if (videoProcessor != null && videoEncoder != null) {

        val processedFrame = videoProcessor.processFrame(imageProxy, faces, isFrontFacing)

        // Log the processed frame details
        Log.d(TAG, "Processed frame: ${processedFrame.width}x${processedFrame.height} (isFrontFacing: ${isFrontFacing})")

        // Draw to surface and encode
        videoProcessor.drawToSurface(processedFrame, videoEncoder.inputSurface)
        videoEncoder.drainEncoder(false)

        Log.d(TAG, "Frame drawn to surface and encoded")
      } else {
        Log.w(TAG, "VideoProcessor or VideoEncoder is null")
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error processing detected faces: ${e.message}")
      e.printStackTrace()
    }
  }

  @SuppressLint("UnsafeOptInUsageError")
  private fun detectFaces(imageProxy: ImageProxy) {
    val mediaImage = imageProxy.image
    if (mediaImage != null) {
      val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
      detector.process(image)
        .addOnSuccessListener { faces ->
          processDetectedFaces(faces, imageProxy)
        }
        .addOnFailureListener { e ->
          Log.e(TAG, "Face analysis failure.", e)
        }
        .addOnCompleteListener {
          imageProxy.close()
        }
    } else {
      imageProxy.close()
    }
  }

  companion object {
    private const val TAG = "FrameProcessorPipeline"
    private const val MINIMUM_PROCESSING_INTERVAL_MS = 33 // Approximately 30 fps
  }
}
