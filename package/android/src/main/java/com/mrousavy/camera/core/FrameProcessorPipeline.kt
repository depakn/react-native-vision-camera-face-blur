package com.mrousavy.camera.core

import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.mrousavy.camera.frameprocessors.Frame

class FrameProcessorPipeline(
  private val callback: CameraSession.Callback,
  private val faceDetectionRecorder: FaceDetectionRecorder,
  private val shouldBlurFace: Boolean
) : ImageAnalysis.Analyzer {

  @OptIn(ExperimentalGetImage::class)
  override fun analyze(imageProxy: ImageProxy) {
    Log.d("FrameProcessorPipeline", "Received a new frame for analysis.")

    val frame = Frame(imageProxy)
    try {
      frame.incrementRefCount()
      callback.onFrame(frame)

      if (shouldBlurFace) {
        Log.d("FrameProcessorPipeline", "Processing frame for face blur.")
        faceDetectionRecorder.processFrame(frame, imageProxy.imageInfo.rotationDegrees)
      }

      Log.d("FrameProcessorPipeline", "Frame processed successfully.")
    } catch (e: Exception) {
      Log.e("FrameProcessorPipeline", "Error processing frame: ${e.message}", e)
    } finally {
      frame.decrementRefCount()
      Log.d("FrameProcessorPipeline", "Frame reference decremented and closed.")
    }
  }
}
