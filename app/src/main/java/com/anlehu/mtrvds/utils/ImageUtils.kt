package com.anlehu.mtrvds.utils

import android.graphics.Matrix
import android.util.Log
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

            var matrix = Matrix()

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
            var transpose: Boolean = (abs(rotation) + 90) % 180 == 0

            var width: Int = if(transpose) previewHeight else previewWidth
            var height: Int = if(transpose) previewWidth else previewHeight

            // Apply scaling if needed
            if (width != destinationWidth || height != destinationHeight){
                val scaleX: Float = destinationWidth / width.toFloat()
                val scaleY: Float = destinationHeight / destinationHeight.toFloat()

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

        // Value for RGB clamping (2^18 - 1) before normalization to 8 bits
        var kMaxChannelValue = 262143


    }

}