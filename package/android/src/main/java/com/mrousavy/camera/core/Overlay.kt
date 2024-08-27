package com.mrousavy.camera.core

import android.content.Context
import android.content.res.Configuration
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.util.Size
import android.view.View
import com.google.mlkit.vision.face.Face

class Overlay @JvmOverloads constructor(
  context: Context,
  attrs: AttributeSet? = null,
  defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {
  private var previewWidth: Int = 0
  private var previewHeight: Int = 0
  private var faces = emptyList<Face>()
  private var isFrontFacing = true
  private var orientation = Configuration.ORIENTATION_PORTRAIT
  private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
    color = Color.YELLOW
    style = Paint.Style.STROKE
    strokeWidth = 5.0f
  }

  override fun onDraw(canvas: Canvas) {
    super.onDraw(canvas)
    faces.forEach { face ->
      drawFaceBorder(face, canvas)
    }
  }

  fun setOrientation(orientation: Int) {
    this.orientation = orientation
  }

  fun setIsFrontFacing(isFrontFacing: Boolean) {
    this.isFrontFacing = isFrontFacing
  }

  fun setPreviewSize(size: Size) {
    previewWidth = size.width
    previewHeight = size.height
  }

  fun setFaces(faceList: List<Face>) {
    faces = faceList
    postInvalidate()
  }

  private fun drawFaceBorder(face: Face, canvas: Canvas) {
    val bounds = face.boundingBox
    val scaledBounds = RectF(
      translateX(bounds.left.toFloat()),
      translateY(bounds.top.toFloat()),
      translateX(bounds.right.toFloat()),
      translateY(bounds.bottom.toFloat())
    )

    canvas.drawRect(scaledBounds, paint)
  }

  private fun translateX(x: Float): Float {
    val scale = width.toFloat() / if (orientation == Configuration.ORIENTATION_LANDSCAPE) previewWidth else previewHeight
    val scaledX = x * scale
    return if (isFrontFacing) width - scaledX else scaledX
  }

  private fun translateY(y: Float): Float {
    val scale = height.toFloat() / if (orientation == Configuration.ORIENTATION_LANDSCAPE) previewHeight else previewWidth
    return y * scale
  }
}
