package com.mrousavy.camera.core

import android.media.MediaRecorder
import android.util.Log
import java.io.File
import java.io.IOException

class AudioRecorder(private val outputFile: File) {
  private var mediaRecorder: MediaRecorder? = null

  fun start() {
    mediaRecorder = MediaRecorder().apply {
      setAudioSource(MediaRecorder.AudioSource.MIC)
      setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
      setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
      setOutputFile(outputFile.absolutePath)

      try {
        prepare()
      } catch (e: IOException) {
        Log.e("AudioRecorder", "prepare() failed: ${e.message}")
      }

      try {
        start()
      } catch (e: IllegalStateException) {
        Log.e("AudioRecorder", "start() failed: ${e.message}")
      }
    }
  }

  fun stop() {
    try {
      mediaRecorder?.apply {
        stop()
        release()
      }
    } catch (e: Exception) {
      Log.e("AudioRecorder", "stop() failed: ${e.message}")
    }
    mediaRecorder = null
  }
}
