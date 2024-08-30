#include <jni.h>
#include <algorithm>
#include <cstring>

const int ALPHA_CHANNEL = 24;
const int RED_CHANNEL = 16;
const int GREEN_CHANNEL = 8;
const int BLUE_CHANNEL = 0;

void stackBlur(int* pix, int w, int h, int radius) {
    // Create working buffer
    int* buffer = new int[w * h];
    std::memcpy(buffer, pix, w * h * sizeof(int));

    int wm = w - 1;
    int hm = h - 1;
    int div = radius * 2 + 1;
    int divsum = (div + 1) >> 1;
    divsum *= divsum;

    int* vmin = new int[std::max(w, h)];
    int* vmax = new int[std::max(w, h)];

    // Apply blur horizontally
    for (int y = 0; y < h; y++) {
        int sum[4] = {0};
        for (int i = -radius; i <= radius; i++) {
            int x = std::min(wm, std::max(i, 0));
            int pixel = buffer[y * w + x];
            sum[0] += (pixel >> ALPHA_CHANNEL) & 0xFF;
            sum[1] += (pixel >> RED_CHANNEL) & 0xFF;
            sum[2] += (pixel >> GREEN_CHANNEL) & 0xFF;
            sum[3] += pixel & 0xFF;
        }
        for (int x = 0; x < w; x++) {
            pix[y * w + x] =
                    ((sum[0] / div) << ALPHA_CHANNEL) |
                    ((sum[1] / div) << RED_CHANNEL) |
                    ((sum[2] / div) << GREEN_CHANNEL) |
                    (sum[3] / div);

            if (x == 0) {
                vmin[y] = std::min(x + radius + 1, wm);
                vmax[y] = std::max(x - radius, 0);
            } else {
                vmin[y] = x + radius > wm ? wm : x + radius;
                vmax[y] = x - radius - 1 < 0 ? 0 : x - radius - 1;
            }

            int p1 = buffer[y * w + vmin[y]];
            int p2 = buffer[y * w + vmax[y]];

            sum[0] += ((p1 >> ALPHA_CHANNEL) & 0xFF) - ((p2 >> ALPHA_CHANNEL) & 0xFF);
            sum[1] += ((p1 >> RED_CHANNEL) & 0xFF) - ((p2 >> RED_CHANNEL) & 0xFF);
            sum[2] += ((p1 >> GREEN_CHANNEL) & 0xFF) - ((p2 >> GREEN_CHANNEL) & 0xFF);
            sum[3] += (p1 & 0xFF) - (p2 & 0xFF);
        }
    }

    // Apply blur vertically
    for (int x = 0; x < w; x++) {
        int sum[4] = {0};
        for (int i = -radius; i <= radius; i++) {
            int y = std::min(hm, std::max(i, 0));
            int pixel = pix[y * w + x];
            sum[0] += (pixel >> ALPHA_CHANNEL) & 0xFF;
            sum[1] += (pixel >> RED_CHANNEL) & 0xFF;
            sum[2] += (pixel >> GREEN_CHANNEL) & 0xFF;
            sum[3] += pixel & 0xFF;
        }
        for (int y = 0; y < h; y++) {
            buffer[y * w + x] =
                    ((sum[0] / div) << ALPHA_CHANNEL) |
                    ((sum[1] / div) << RED_CHANNEL) |
                    ((sum[2] / div) << GREEN_CHANNEL) |
                    (sum[3] / div);

            if (y == 0) {
                vmin[x] = std::min(y + radius + 1, hm);
                vmax[x] = std::max(y - radius, 0);
            } else {
                vmin[x] = y + radius > hm ? hm : y + radius;
                vmax[x] = y - radius - 1 < 0 ? 0 : y - radius - 1;
            }

            int p1 = pix[vmin[x] * w + x];
            int p2 = pix[vmax[x] * w + x];

            sum[0] += ((p1 >> ALPHA_CHANNEL) & 0xFF) - ((p2 >> ALPHA_CHANNEL) & 0xFF);
            sum[1] += ((p1 >> RED_CHANNEL) & 0xFF) - ((p2 >> RED_CHANNEL) & 0xFF);
            sum[2] += ((p1 >> GREEN_CHANNEL) & 0xFF) - ((p2 >> GREEN_CHANNEL) & 0xFF);
            sum[3] += (p1 & 0xFF) - (p2 & 0xFF);
        }
    }

    // Copy result back to pix
    std::memcpy(pix, buffer, w * h * sizeof(int));

    // Clean up
    delete[] buffer;
    delete[] vmin;
    delete[] vmax;
}

extern "C" JNIEXPORT void JNICALL
Java_com_mrousavy_camera_core_FaceDetectionRecorder_nativeStackBlur(
        JNIEnv* env,
jobject /* this */,
jintArray pixelsArray,
        jint width,
jint height,
        jint radius) {
jint* pixels = env->GetIntArrayElements(pixelsArray, nullptr);

stackBlur(pixels, width, height, radius);

env->ReleaseIntArrayElements(pixelsArray, pixels, 0);
}
