package com.anlehu.mmhcods.utils

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Environment
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import kotlin.math.abs
import kotlin.math.max

class ImageUtils {

    companion object{
        fun getTransformationMatrix(
            previewWidth: Int,
            previewHeight: Int,
            destinationWidth: Int,
            destinationHeight: Int,
            rotation: Int,
            maintainAspectRatio: Boolean): Matrix {

            val matrix = Matrix()

            if(rotation !=0 ){
                if(rotation % 90 != 0){
                    Log.w(Logger.DEF_TAG, "Rotation is not multiple of 90")
                }
                // Center of image at origin
                matrix.postTranslate(-previewWidth / 2.0f, -previewHeight / 2.0f)
                // Rotate around origin
                matrix.postRotate(rotation.toFloat())
            }

            // Scale rotation
            val transpose: Boolean = (abs(rotation) + 90) % 180 == 0

            val width: Int = if(transpose) previewHeight else previewWidth
            val height: Int = if(transpose) previewWidth else previewHeight

            // Apply scaling if needed
            if (width != destinationWidth || height != destinationHeight){
                val scaleX: Float = destinationWidth / width.toFloat()
                val scaleY: Float = destinationHeight / height.toFloat()

                if (maintainAspectRatio){
                    val scaleFactor = max(scaleX, scaleY)
                    matrix.postScale(scaleFactor, scaleFactor)
                }
                else{
                    matrix.postScale(scaleX, scaleY)
                }
            }

            // Translate back to origin centered reference to destination frame
            if (rotation != 0){
                matrix.postTranslate(destinationWidth / 2.0f, destinationHeight / 2.0f)
            }

            return matrix

        }

        fun normalize(bitmap: Bitmap){

        }

        fun saveBitmap(bitmap: Bitmap, filename: String) {
            val root = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM).absolutePath
            Log.i("Saving bitmap to", "${bitmap.width}, ${bitmap.height}, $root")
            val myDir = File(root)
            if (!myDir.mkdirs()) {
                Log.e("ERROR", "MKDIR FAILED")
            }
            val file = File(myDir, filename)
            if (file.exists()) {
                file.delete()
            }
            try {
                val out = FileOutputStream(file)
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                out.flush()
                out.close()
            } catch (e: Exception) {
                e.toString()
            }
        }

        fun convertYUVtoARGB(
            yBytes: ByteArray,
            uBytes: ByteArray,
            vBytes: ByteArray,
            screenPreviewWidth: Int,
            screenPreviewHeight: Int,
            yRowStride: Int,
            uvRowStride: Int,
            uvPixelStride: Int,
            rgbBytes: IntArray) {

            Log.d("CONVERTING:", "$screenPreviewWidth x $screenPreviewHeight")
            var yPixel = 0
            for(i in 0 until screenPreviewHeight){
                val pixelY = yRowStride * i
                val pixelUV = uvRowStride * (i shr 1)

                for(j in 0 until screenPreviewWidth){
                    var offset = pixelUV + (j shr 1) * uvPixelStride
                    rgbBytes[yPixel++] = YUVTORGB(0xff and yBytes[pixelY+j].toInt(), 0xff and uBytes[offset].toInt(), 0xff and vBytes[offset].toInt())
                }
            }



        }

        private fun YUVTORGB(yVal: Int, uVal: Int, vVal: Int): Int {

            // Adjust and check YUV values
            var y = if (yVal - 16 < 0) 0 else yVal - 16
            var u = uVal - 128
            var v = vVal - 128

            y *= 1192
            var r = y + 1634 * v
            var g = y - 833 * v - 400 * u
            var b = y + 2066 * u

            // Clip RGB values within 0 to kMaxChannelValue

            // Clipping RGB values to be inside boundaries [ 0 , kMaxChannelValue ]
            r = if (r > kMaxChannelValue) kMaxChannelValue else if (r < 0) 0 else r
            g = if (g > kMaxChannelValue) kMaxChannelValue else if (g < 0) 0 else g
            b = if (b > kMaxChannelValue) kMaxChannelValue else if (b < 0) 0 else b

            return -0x1000000 or (r shl 6 and 0xff0000) or (g shr 2 and 0xff00) or (b shr 10 and 0xff)
        }

        // Value for RGB clamping (2^18 - 1) before normalization to 8 bits
        var kMaxChannelValue = 262143


    }

}