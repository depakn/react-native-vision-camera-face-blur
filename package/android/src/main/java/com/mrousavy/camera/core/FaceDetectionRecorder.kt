package com.mrousavy.camera.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.media.Image
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.util.Log
import android.util.Size
import android.view.Surface
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.mrousavy.camera.frameprocessors.Frame
import java.io.File
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class FaceDetectionRecorder(private val context: Context) {
  private enum class State {
    IDLE,
    STARTED,
    STOPPING,
    STOPPED
  }

  private var encoder: MediaCodec? = null
  private var muxer: MediaMuxer? = null
  private var inputSurface: Surface? = null
  private var trackIndex = -1
  private var muxerStarted = false
  private var frameCount: Long = 0

  private val options = FaceDetectorOptions.Builder()
    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
    .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
    .setMinFaceSize(0.15f)
    .build()
  private val faceDetector = FaceDetection.getClient(options)
  private val processingExecutor: Executor = Executors.newFixedThreadPool(3)

  private var frameSize: Size = Size(1, 1)
  private val state = AtomicReference(State.IDLE)

  private var audioRecorder: AudioRecorder? = null
  private var isFrontCamera: Boolean = false

  private val lock = ReentrantLock()

  private var lastProcessedFrameTime = 0L
  private val PROCESS_INTERVAL_MS = 100 // Process every 100ms

  init {
    System.loadLibrary("VisionCamera")
  }

  private external fun nativeYUV420toARGB8888(yuv: ByteArray, width: Int, height: Int, out: IntArray)

  private external fun nativeStackBlur(pix: IntArray, w: Int, h: Int, radius: Int)

  fun startRecording(outputFile: File, processedAudioFile: File, size: Size, isFrontCamera: Boolean) {
    lock.withLock {
      this.isFrontCamera = isFrontCamera
      if (state.get() != State.IDLE) {
        Log.w(TAG, "Cannot start recording. Current state: ${state.get()}")
        return
      }

      frameSize = size
      state.set(State.STARTED)
      Log.d("FaceDetectionRecorder", "MediaFormat Width: ${size.width} ${size.height}")
      try {
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, size.width, size.height)
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
        format.setInteger(MediaFormat.KEY_BIT_RATE, 10_000_000)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)

        encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC).apply {
          configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
          inputSurface = createInputSurface()
          start()
        }

        muxer = MediaMuxer(outputFile.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)

        audioRecorder = AudioRecorder(processedAudioFile)
        audioRecorder?.start()

        Log.i(TAG, "Recording started successfully")
      } catch (e: Exception) {
        Log.e(TAG, "Error starting recording", e)
        stopRecording()
      }
    }
  }

  fun processFrame(frame: Frame, rotationDegrees: Int) {
    if (state.get() != State.STARTED) return

    val currentTime = System.currentTimeMillis()
    if (currentTime - lastProcessedFrameTime < PROCESS_INTERVAL_MS) return

    frame.incrementRefCount()
    processingExecutor.execute {
      try {
        lock.withLock {
          if (state.get() != State.STARTED) {
            frame.decrementRefCount()
            return@execute
          }

          val image = frame.image
          if (image != null) {
            val bitmap = imageToBitmap(image)
            val inputImage = InputImage.fromMediaImage(image, rotationDegrees)

            faceDetector.process(inputImage)
              .addOnSuccessListener { faces ->
                lock.withLock {
                  if (state.get() == State.STARTED) {
                    val rotatedBitmap = rotateBitmap(bitmap, rotationDegrees)
                    drawFacesAndBitmapToSurface(rotatedBitmap, faces, rotationDegrees)
                    drainEncoder(false)
                    rotatedBitmap.recycle()
                  }
                }
              }
              .addOnFailureListener { e -> Log.e(TAG, "Face detection failed", e) }
              .addOnCompleteListener {
                bitmap.recycle()
                frame.decrementRefCount()
              }
          } else {
            frame.decrementRefCount()
          }
        }
      } catch (e: Exception) {
        Log.e(TAG, "Error processing frame", e)
        frame.decrementRefCount()
      }
    }
    lastProcessedFrameTime = currentTime
  }

  private fun imageToBitmap(image: Image): Bitmap {
    val yuvBytes = imageToByteArray(image)
    val argbArray = IntArray(image.width * image.height)
    nativeYUV420toARGB8888(yuvBytes, image.width, image.height, argbArray)
    return Bitmap.createBitmap(argbArray, image.width, image.height, Bitmap.Config.ARGB_8888)
  }

  private fun imageToByteArray(image: Image): ByteArray {
    val yBuffer = image.planes[0].buffer
    val uBuffer = image.planes[1].buffer
    val vBuffer = image.planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    return nv21
  }

  private fun drawFacesAndBitmapToSurface(bitmap: Bitmap, faces: List<Face>, rotationDegrees: Int) {
    try {
      inputSurface?.let { surface ->
        val canvas = surface.lockCanvas(null)

        // Clear the canvas with a black background
        canvas.drawColor(Color.BLACK)

        Log.d("FaceDetectionRecorder", "Scale Factors: ${frameSize.width} - ${bitmap.width}")
        // Calculate scaling factors
        val scaleX = frameSize.width.toFloat() / bitmap.width
        val scaleY = frameSize.height.toFloat() / bitmap.height
        val scale = minOf(scaleX, scaleY)

        // Calculate translation to center the bitmap
        val translateX = (frameSize.width - bitmap.width * scale) / 2f
        val translateY = (frameSize.height - bitmap.height * scale) / 2f

        // Apply transformations
        canvas.translate(translateX, translateY)
        canvas.scale(scale, scale)

        if (isFrontCamera) {
          canvas.scale(-1f, 1f)
          canvas.translate(-bitmap.width.toFloat(), 0f)
        }

        // Draw the bitmap
        canvas.drawBitmap(bitmap, 0f, 0f, null)

        val paint = Paint().apply {
          color = Color.rgb(255, 224, 196)
          style = Paint.Style.FILL_AND_STROKE
          strokeWidth = 5f / scale
        }

        // Blur face regions
        faces.forEach { face ->
          val rect = face.boundingBox
          val blurredRegion = blurBitmapRegion(bitmap, rect)
          canvas.drawBitmap(blurredRegion, rect.left.toFloat(), rect.top.toFloat(), null)
          blurredRegion.recycle()
//          canvas.drawRect(rect, paint)
        }

        surface.unlockCanvasAndPost(canvas)
      }
    } catch (ex: Exception) {
      Log.e(TAG, "Error in drawFacesAndBitmapToSurface", ex)
    }
  }

  private fun blurBitmapRegion(source: Bitmap, region: Rect): Bitmap {
    // Downscale region
    val downscaleFactor = 0.5
    val downscaledWidth = (region.width() * downscaleFactor).toInt()
    val downscaledHeight = (region.height() * downscaleFactor).toInt()

    val scaledRegion = Bitmap.createScaledBitmap(
      source, downscaledWidth, downscaledHeight, true
    )

    // Apply blur to the downscaled region
    val pixels = IntArray(scaledRegion.width * scaledRegion.height)
    scaledRegion.getPixels(pixels, 0, scaledRegion.width, 0, 0, scaledRegion.width, scaledRegion.height)
    nativeStackBlur(pixels, scaledRegion.width, scaledRegion.height, 20)

    // Scale it back to original size
    val blurredBitmap = Bitmap.createBitmap(pixels, scaledRegion.width, scaledRegion.height, Bitmap.Config.ARGB_8888)
    return Bitmap.createScaledBitmap(blurredBitmap, region.width(), region.height(), true)
  }

  private fun rotateBitmap(source: Bitmap, angle: Int): Bitmap {
    val matrix = Matrix()
    matrix.postRotate(angle.toFloat())
    return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
  }

  private fun rotateRect(rect: Rect, width: Int, height: Int, rotation: Int): RectF {
    val matrix = Matrix()
    matrix.setRotate(rotation.toFloat(), width / 2f, height / 2f)

    val rotatedRect = RectF(rect)
    matrix.mapRect(rotatedRect)

    return rotatedRect
  }

  private fun drainEncoder(endOfStream: Boolean) {
    val encoder = encoder ?: return
    if (endOfStream) {
      encoder.signalEndOfInputStream()
    }

    val bufferInfo = MediaCodec.BufferInfo()
    while (true) {
      if (state.get() != State.STARTED && state.get() != State.STOPPING) {
        break
      }
      val encoderStatus = encoder.dequeueOutputBuffer(bufferInfo, TIMEOUT_USEC)
      if (encoderStatus == MediaCodec.INFO_TRY_AGAIN_LATER) {
        if (!endOfStream) break
      } else if (encoderStatus == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
        if (muxerStarted) throw RuntimeException("Format changed twice")
        val newFormat = encoder.outputFormat
        trackIndex = muxer?.addTrack(newFormat) ?: -1
        muxer?.start()
        muxerStarted = true
      } else if (encoderStatus < 0) {
        Log.w(TAG, "Unexpected result from encoder.dequeueOutputBuffer: $encoderStatus")
      } else {
        val encodedData = encoder.getOutputBuffer(encoderStatus) ?: continue
        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) {
          bufferInfo.size = 0
        }

        if (bufferInfo.size != 0) {
          if (!muxerStarted) throw RuntimeException("muxer hasn't started")
          encodedData.position(bufferInfo.offset)
          encodedData.limit(bufferInfo.offset + bufferInfo.size)
          muxer?.writeSampleData(trackIndex, encodedData, bufferInfo)
        }

        encoder.releaseOutputBuffer(encoderStatus, false)

        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
          break
        }
      }
    }
  }

  fun stopRecording() {
    lock.withLock {
      if (state.get() != State.STARTED) {
        Log.w(TAG, "Cannot stop recording. Current state: ${state.get()}")
        return
      }

      state.set(State.STOPPING)
    }

    processingExecutor.execute {
      lock.withLock {
        try {
          drainEncoder(true)
          encoder?.stop()
          encoder?.release()
          audioRecorder?.stop()
          muxer?.stop()
          muxer?.release()
          inputSurface?.release()
        } catch (e: Exception) {
          Log.e(TAG, "Error stopping recording", e)
        } finally {
          encoder = null
          muxer = null
          audioRecorder = null
          inputSurface = null
          state.set(State.STOPPED)
        }
      }
    }

    (processingExecutor as java.util.concurrent.ExecutorService).shutdown()
  }

  companion object {
    private const val TAG = "FaceDetectionRecorder"
    private const val TIMEOUT_USEC = 10000L
  }
}
