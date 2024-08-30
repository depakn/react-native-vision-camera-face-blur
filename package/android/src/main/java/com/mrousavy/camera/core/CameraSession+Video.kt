package com.mrousavy.camera.core

import android.annotation.SuppressLint
import android.util.Log
import android.util.Size
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.video.ExperimentalPersistentRecording
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.VideoRecordEvent
import com.mrousavy.camera.core.extensions.getCameraError
import com.mrousavy.camera.core.types.Orientation
import com.mrousavy.camera.core.types.RecordVideoOptions
import com.mrousavy.camera.core.types.Video
import java.io.File

@OptIn(ExperimentalPersistentRecording::class)
@SuppressLint("MissingPermission", "RestrictedApi")
fun CameraSession.startRecording(
  enableAudio: Boolean,
  options: RecordVideoOptions,
  callback: (video: Video) -> Unit,
  onError: (error: CameraError) -> Unit
) {
  if (camera == null) throw CameraNotReadyError()
  if (recording != null) throw RecordingInProgressError()
  val videoOutput = videoOutput ?: throw VideoNotEnabledError()

  // Create output video file
  val outputOptions = FileOutputOptions.Builder(options.file.file).also { outputOptions ->
    metadataProvider.location?.let { location ->
      Log.i(CameraSession.TAG, "Setting Video Location to ${location.latitude}, ${location.longitude}...")
      outputOptions.setLocation(location)
    }
  }.build()

  // Prepare recording
  var pendingRecording = videoOutput.output.prepareRecording(context, outputOptions)
  if (enableAudio) {
    checkMicrophonePermission()
    pendingRecording = pendingRecording.withAudioEnabled()
  }
  pendingRecording = pendingRecording.asPersistentRecording()

  val timestampMS = System.currentTimeMillis()
  val processedVideoFile = File(context.cacheDir, "processed_video_${timestampMS}.mp4")
  val processedAudioFile = File(context.cacheDir, "processed_audio_${timestampMS}.m4a")
  val finalProcessedFile = File(context.cacheDir, "final_video_${timestampMS}.mp4")

  isRecordingCanceled = false
  var hasReceivedFrames = false
  recording = pendingRecording.start(CameraQueues.cameraExecutor) { event ->
    when (event) {
      is VideoRecordEvent.Start -> {
        Log.i(CameraSession.TAG, "Recording started!")

        if (configuration?.shouldBlurFace == true) {
          hasReceivedFrames = false

          val width = configuration?.format?.videoWidth ?: 640
          val height = configuration?.format?.videoHeight ?: 1280

          val isFrontCamera = camera?.cameraInfo?.lensFacing == CameraSelector.LENS_FACING_FRONT

          if (outputOrientation == Orientation.PORTRAIT || outputOrientation == Orientation.PORTRAIT_UPSIDE_DOWN) {
            faceDetectionRecorder.startRecording(processedVideoFile, processedAudioFile, Size(height, width), isFrontCamera)
          } else {
            faceDetectionRecorder.startRecording(processedVideoFile, processedAudioFile, Size(width, height), isFrontCamera)
          }
        }
      }

      is VideoRecordEvent.Resume -> Log.i(CameraSession.TAG, "Recording resumed!")

      is VideoRecordEvent.Pause -> Log.i(CameraSession.TAG, "Recording paused!")

      is VideoRecordEvent.Status -> {
        Log.i(CameraSession.TAG, "Status update! Recorded ${event.recordingStats.numBytesRecorded} bytes.")
        if (event.recordingStats.numBytesRecorded > 0) {
          hasReceivedFrames = true
        }
      }

      is VideoRecordEvent.Finalize -> {
        if (isRecordingCanceled) {
          Log.i(CameraSession.TAG, "Recording was canceled, deleting file..")
          onError(RecordingCanceledError())
          try {
            options.file.file.delete()
          } catch (e: Throwable) {
            this.callback.onError(FileIOError(e))
          }
          return@start
        }

        if (configuration?.shouldBlurFace == true) {
          if (!hasReceivedFrames) {
            Log.e(CameraSession.TAG, "Recording stopped before receiving any frames.")
            return@start
          }
        }

        Log.i(CameraSession.TAG, "Recording stopped!")
        val error = event.getCameraError()
        if (error != null) {
          if (error.wasVideoRecorded) {
            Log.e(CameraSession.TAG, "Video Recorder encountered an error, but the video was recorded anyways.", error)
          } else {
            Log.e(CameraSession.TAG, "Video Recorder encountered a fatal error!", error)
            onError(error)
            return@start
          }
        }

        // Prepare output result
        val durationMs = event.recordingStats.recordedDurationNanos / 1_000_000
        Log.i(CameraSession.TAG, "Successfully completed video recording! Captured ${durationMs.toDouble() / 1_000.0} seconds.")

        if (configuration?.shouldBlurFace == true) {
          // Merge Audio and Video
          mergeAudioVideo(processedVideoFile, processedAudioFile, finalProcessedFile)

          val size = videoOutput.attachedSurfaceResolution ?: Size(0, 0)
          val video = Video(finalProcessedFile.absolutePath, durationMs, size)
          callback(video)

          processedVideoFile.delete();
          processedAudioFile.delete();
        } else {
          val path = event.outputResults.outputUri.path ?: throw UnknownRecorderError(false, null)
          val size = videoOutput.attachedSurfaceResolution ?: Size(0, 0)
          val video = Video(path, durationMs, size)
          callback(video)
        }
      }
    }
  }
}

private fun mergeAudioVideo(videoFile: File, audioFile: File, outputFile: File) {
  try {
    val command = arrayOf(
      "-i", videoFile.absolutePath,
      "-i", audioFile.absolutePath,
      "-c:v", "copy",
      "-c:a", "aac",
      "-strict", "experimental",
      outputFile.absolutePath
    )

    com.arthenica.mobileffmpeg.FFmpeg.execute(command)

    Log.d("RnFaceBlurViewManager", "Audio and video merged successfully")
  } catch (e: Exception) {
    Log.e("RnFaceBlurViewManager", "Error merging audio and video: ${e.message}")
  }
}

fun CameraSession.stopRecording() {
  val recording = recording ?: throw NoRecordingInProgressError()

  CameraQueues.cameraExecutor.execute {
    if (configuration?.shouldBlurFace == true) {
      faceDetectionRecorder.stopRecording()
    }
    Thread.sleep(500)
    recording.stop()
    this.recording = null
  }
}

fun CameraSession.cancelRecording() {
  isRecordingCanceled = true
  stopRecording()
}

fun CameraSession.pauseRecording() {
  val recording = recording ?: throw NoRecordingInProgressError()
  recording.pause()
}

fun CameraSession.resumeRecording() {
  val recording = recording ?: throw NoRecordingInProgressError()
  recording.resume()
}
