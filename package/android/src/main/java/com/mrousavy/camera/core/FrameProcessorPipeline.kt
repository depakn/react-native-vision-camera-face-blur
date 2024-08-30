package com.mrousavy.camera.core

import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis.Analyzer
import androidx.camera.core.ImageProxy
import com.mrousavy.camera.frameprocessors.Frame

class FrameProcessorPipeline(
  private val callback: CameraSession.Callback,
  private val faceDetectionRecorder: FaceDetectionRecorder,
  private val shouldBlurFace: Boolean
) : Analyzer {
  @OptIn(ExperimentalGetImage::class)
  override fun analyze(imageProxy: ImageProxy) {
    val frame = Frame(imageProxy)
    try {
      frame.incrementRefCount()
      callback.onFrame(frame)
      if (shouldBlurFace) {
        faceDetectionRecorder.processFrame(frame, imageProxy.imageInfo.rotationDegrees)
      }
    } finally {
      frame.decrementRefCount()
    }
  }
}
