package com.anlehu.mmhcods.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Environment
import android.util.Log
import android.util.Size
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import kotlin.math.abs
import kotlin.math.max

class ImageUtils {

    companion object{
        /********************************************************************************************************
         * Gets transformation matrix
         ********************************************************************************************************/
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
        /********************************************************************************************************
         * Saves image to device
         ********************************************************************************************************/
        fun saveBitmap(bitmap: Bitmap, filename: String) {
            val root = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM).absolutePath
            Log.i("Saving bitmap to", "${bitmap.width}, ${bitmap.height}, $root")
            val myDir = File(root)
            if (!myDir.mkdirs()) {
                Log.e("ERROR", "MKDIR FAILED")
            }
            val file = File(myDir, filename+".png")
            if (file.exists()) {
                file.delete()
                Log.d("DEL", "FILE DELETED")
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

        /********************************************************************************************************
         * Resizes image to fit model input dimensions while adding padding
         ********************************************************************************************************/
        fun rescaleImage(inpBitmap: Bitmap?, size: Size, path: String, rotate: Float, pad: Boolean): Bitmap{

            var bitmap: Bitmap = Bitmap.createBitmap(inpBitmap!!)
            var fileName: String

            if(pad){
                if(inpBitmap == null){
                    var opt = BitmapFactory.Options()
                    opt.inPreferredConfig = Bitmap.Config.ARGB_8888
                    bitmap = BitmapFactory.decodeFile(path, opt)
                    fileName = path.substringBeforeLast('.').substring(path.substringBeforeLast('.').length - 4)
                }else{
                    bitmap = inpBitmap
                    fileName = "Test Bitmap"
                }

                val oldHeight = bitmap.height
                val oldWidth = bitmap.width

                val ratio = (640.0 / max(oldHeight, oldWidth))
                val newSize = listOf((oldWidth * ratio), (oldHeight*ratio))

                var srcMat = Mat(oldHeight, oldWidth, CvType.CV_8UC4, Scalar(4.0))
                val dstMat = Mat(size.height, size.width, CvType.CV_8UC4, Scalar(4.0))

                Utils.bitmapToMat(bitmap, srcMat)
                Log.d("MAT_SIZE", srcMat.cols().toString()+" ratio: $ratio oldHeight: $oldHeight oldWidth: $oldWidth")
                Imgproc.resize(srcMat, dstMat, org.opencv.core.Size(), ratio, ratio, Imgproc.INTER_AREA)

                val deltaW = 640 - newSize[0]
                val deltaH = 640 - newSize[1]
                val top = deltaH.toInt().floorDiv(2)
                val bottom = deltaH.toInt() - deltaH.toInt().floorDiv(2)
                val left = deltaW.toInt().floorDiv(2)
                val right = deltaW.toInt() - deltaW.toInt().floorDiv(2)

                Core.copyMakeBorder(dstMat, dstMat, top, bottom, left, right, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0, 255.0))
                bitmap = Bitmap.createBitmap(dstMat.width(), dstMat.height(), Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(dstMat, bitmap)
                saveBitmap(bitmap, fileName)
            }else{
                val mat = Matrix()
                mat.postRotate(rotate)
                val rotatedBitmap = Bitmap.createBitmap(inpBitmap, 0, 0, inpBitmap.width, inpBitmap.height, mat, true)
                bitmap = Bitmap.createScaledBitmap(rotatedBitmap, size.width, size.height, true)
            }

            return bitmap
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