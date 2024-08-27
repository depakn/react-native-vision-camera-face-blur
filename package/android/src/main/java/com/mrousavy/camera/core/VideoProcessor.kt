package com.mrousavy.camera.core

import android.graphics.*
import android.util.Log
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.face.Face
import android.view.Surface
import java.io.ByteArrayOutputStream

class VideoProcessor(width: Int, height: Int) {
  private var previewWidth: Int = width
  private var previewHeight: Int = height

  init {
    System.loadLibrary("VisionCamera")
  }

  fun processFrame(image: ImageProxy, faces: List<Face>, isFrontFacing: Boolean): Bitmap {
    val bitmap = image.toBitmap()
    val angle = if (isFrontFacing) -90f else 90f
    val rotatedBitmap = rotateImage(bitmap, angle)

    Log.d("VideoProcessor", "Original image size: ${image.width}x${image.height}")
    Log.d("VideoProcessor", "Rotated bitmap size: ${rotatedBitmap.width}x${rotatedBitmap.height}")

    val scaleX = rotatedBitmap.width.toFloat() / image.height.toFloat()
    val scaleY = rotatedBitmap.height.toFloat() / image.width.toFloat()

    Log.d("VideoProcessor", "Scale factors: scaleX=$scaleX, scaleY=$scaleY")

    val mutableBitmap = rotatedBitmap.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(mutableBitmap)

    faces.forEachIndexed { index, face ->
      val bounds = face.boundingBox
      Log.d("VideoProcessor", "Face #$index original bounds: $bounds")

      val left = (scaleX * bounds.left.toFloat()).toInt().coerceAtLeast(0)
      val top = (scaleY * bounds.top.toFloat()).toInt().coerceAtLeast(0)
      val right = (scaleX * bounds.right.toFloat()).toInt().coerceAtMost(mutableBitmap.width)
      val bottom = (scaleY * bounds.bottom.toFloat()).toInt().coerceAtMost(mutableBitmap.height)

      Log.d("VideoProcessor", "Face #$index transformed bounds: L=$left, T=$top, R=$right, B=$bottom")

      val width = right - left
      val height = bottom - top

      if (width > 0 && height > 0) {
        val faceBitmap = Bitmap.createBitmap(mutableBitmap, left, top, width, height)
        val blurredFace = blurBitmap(faceBitmap)
        canvas.drawBitmap(blurredFace, left.toFloat(), top.toFloat(), null)
        Log.d("VideoProcessor", "Face #$index blurred successfully")
      } else {
        Log.w("VideoProcessor", "Face #$index has invalid dimensions: ${width}x${height}")
      }
    }

    return mutableBitmap
  }

  private fun blurBitmap(source: Bitmap): Bitmap {
    val scaleFactor = 0.25f
    val radius = 10

    val width = (source.width * scaleFactor).toInt()
    val height = (source.height * scaleFactor).toInt()
    val scaledBitmap = Bitmap.createScaledBitmap(source, width, height, true)

    val pixels = IntArray(width * height)
    scaledBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
    nativeStackBlur(pixels, width, height, radius)
    scaledBitmap.setPixels(pixels, 0, width, 0, 0, width, height)

    return Bitmap.createScaledBitmap(scaledBitmap, source.width, source.height, true)
  }

  private external fun nativeStackBlur(pix: IntArray, w: Int, h: Int, radius: Int)

  fun drawToSurface(bitmap: Bitmap, surface: Surface?) {
    surface?.let {
      try {
        val canvas = it.lockCanvas(null)
        canvas.drawColor(Color.BLACK, PorterDuff.Mode.CLEAR)
        canvas.drawBitmap(bitmap, null, Rect(0, 0, previewWidth, previewHeight), null)
        it.unlockCanvasAndPost(canvas)
        Log.d("VideoProcessor", "Frame drawn to surface: ${bitmap.width}x${bitmap.height}")
      } catch (e: Exception) {
        Log.e("VideoProcessor", "Error drawing to surface: ${e.message}")
        e.printStackTrace()
      }
    } ?: Log.e("VideoProcessor", "Surface is null")
  }

  private fun ImageProxy.toBitmap(): Bitmap {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
    val imageBytes = out.toByteArray()
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
  }

  private fun rotateImage(source: Bitmap, angle: Float): Bitmap {
    val matrix = Matrix()
    matrix.postRotate(angle)
    return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
  }
}
