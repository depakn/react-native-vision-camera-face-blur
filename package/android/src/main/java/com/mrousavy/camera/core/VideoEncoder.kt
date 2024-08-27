package com.mrousavy.camera.core

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.view.Surface
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class VideoEncoder(
  private val width: Int,
  private val height: Int,
  private val outputFile: File
) {
  private val TAG = "VideoEncoder"
  private val encoder: MediaCodec
  private val muxer: MediaMuxer
  private var trackIndex: Int = -1
  private var muxerStarted = false
  private val isRunning = AtomicBoolean(true)
  private var frameCount = 0
  val inputSurface: Surface
  private val lock = ReentrantLock()
  private val encoderDone = lock.newCondition()

  init {
    Log.d(TAG, "Initializing VideoEncoder with width: $width, height: $height")
    val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
      setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
      setInteger(MediaFormat.KEY_BIT_RATE, 10_000_000)
      setInteger(MediaFormat.KEY_FRAME_RATE, 30)
      setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
    }

    encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC).apply {
      configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
      inputSurface = createInputSurface()
      start()
    }

    muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

    Log.d(TAG, "VideoEncoder initialized. Output file: ${outputFile.absolutePath}")
  }

  fun drainEncoder(endOfStream: Boolean) {
    val TIMEOUT_USEC = 10000L
    lock.withLock {
      if (endOfStream) {
        Log.d(TAG, "Signaling end of input stream")
        encoder.signalEndOfInputStream()
      }

      var encoderOutputAvailable = true
      var encoderEOS = false

      while (encoderOutputAvailable && !encoderEOS) {
        val bufferInfo = MediaCodec.BufferInfo()
        val outputBufferIndex = encoder.dequeueOutputBuffer(bufferInfo, TIMEOUT_USEC)

        when (outputBufferIndex) {
          MediaCodec.INFO_TRY_AGAIN_LATER -> {
            encoderOutputAvailable = false
            Log.d(TAG, "No output available yet")
          }
          MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
            val newFormat = encoder.outputFormat
            Log.d(TAG, "Encoder output format changed: $newFormat")
            if (muxerStarted) {
              throw RuntimeException("Format changed twice")
            }
            trackIndex = muxer.addTrack(newFormat)
            muxer.start()
            muxerStarted = true
            Log.d(TAG, "Muxer started")
          }
          MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> {
            Log.d(TAG, "Encoder output buffers changed")
          }
          else -> {
            val encodedData = encoder.getOutputBuffer(outputBufferIndex)
            if (encodedData == null) {
              Log.e(TAG, "encoderOutputBuffer $outputBufferIndex was null")
            } else {
              if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                bufferInfo.size = 0
              }

              if (bufferInfo.size != 0) {
                if (!muxerStarted) {
                  throw RuntimeException("muxer hasn't started")
                }

                encodedData.position(bufferInfo.offset)
                encodedData.limit(bufferInfo.offset + bufferInfo.size)
                muxer.writeSampleData(trackIndex, encodedData, bufferInfo)
                Log.d(TAG, "Encoded frame written to muxer. Size: ${bufferInfo.size}")
                frameCount++
              }

              encoder.releaseOutputBuffer(outputBufferIndex, false)

              if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                encoderEOS = true
                Log.d(TAG, "End of stream reached")
              }
            }
          }
        }
      }

      if (endOfStream) {
        encoderDone.signal()
      }
    }
  }

  fun stop() {
    Log.d(TAG, "Stopping VideoEncoder")
    if (isRunning.getAndSet(false)) {
      lock.withLock {
        try {
          drainEncoder(true)
          encoderDone.await(5, TimeUnit.SECONDS) // Wait for encoder to finish
          encoder.stop()
          encoder.release()
          muxer.stop()
          muxer.release()
          Log.d(TAG, "Encoder stopped and released")
          Log.d(TAG, "Video saved to: ${outputFile.absolutePath}")
          Log.d(TAG, "Total frames processed: $frameCount")
          if (outputFile.exists()) {
            Log.d(TAG, "Output file size: ${outputFile.length()} bytes")
          } else {
            Log.e(TAG, "Output file does not exist!")
          }
        } catch (e: Exception) {
          Log.e(TAG, "Error stopping encoder: ${e.message}")
          e.printStackTrace()
        }
      }
    }
  }
}
