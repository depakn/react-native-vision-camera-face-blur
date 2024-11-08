#include <algorithm>
#include <android/bitmap.h>
#include <cstdint>
#include <jni.h>

// Clamp a value to the range 0-255
inline uint8_t clamp(int x) {
  return (x < 0) ? 0 : ((x > 255) ? 255 : static_cast<uint8_t>(x));
}

extern "C" JNIEXPORT void JNICALL Java_com_mrousavy_camera_core_FaceDetectionRecorder_nativeYUV420toARGB8888(
    JNIEnv* env, jobject /* this */, jbyteArray yuv420sp, jint width, jint height, jintArray rgbOut) {

  jbyte* yuv = env->GetByteArrayElements(yuv420sp, NULL);
  jint* rgb = env->GetIntArrayElements(rgbOut, NULL);

  int frameSize = width * height;
  int j = 0;
  int yp = 0;
  for (int i = 0; i < height; i++) {
    int uvp = frameSize + (i >> 1) * width;
    int u = 0;
    int v = 0;

    for (int x = 0; x < width; x++) {
      int y = (0xff & yuv[yp]) - 16;
      if (y < 0)
        y = 0;

      if ((x & 1) == 0) {
        v = (0xff & yuv[uvp++]) - 128;
        u = (0xff & yuv[uvp++]) - 128;
      }

      int y1192 = 1192 * y;
      int r = (y1192 + 1634 * v);
      int g = (y1192 - 833 * v - 400 * u);
      int b = (y1192 + 2066 * u);

      r = clamp((r >> 10) + 2);
      g = clamp((g >> 10) + 2);
      b = clamp((b >> 10) + 2);

      rgb[yp] = 0xff000000 | (r << 16) | (g << 8) | b;

      yp++;
    }
  }

  env->ReleaseByteArrayElements(yuv420sp, yuv, JNI_ABORT);
  env->ReleaseIntArrayElements(rgbOut, rgb, 0);
}