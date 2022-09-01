package com.anlehu.mmhcods.utils

import android.graphics.RectF
import java.io.File
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ModelUtils {
    companion object{

        var MINIMUM_CONFIDENCE = 0.4f
        var IS_QUANTIZED = false
        var MAIN_MODEL_NAME: String = "main_detector_trike.tflite"
        var LANE_MODEL_NAME: String = "lane_detector.tflite"
        var MAIN_LABELS_NAME: String = "main_labels.txt"
        var MAINTAIN_ASPECT = true

        @Throws(IOException::class)
        fun loadModelFile(mountedPath: String?, modelFilename: String?): MappedByteBuffer? {
            lateinit var file: File
            try{
                file = File(mountedPath+"/"+modelFilename!!)
            }catch(e: Exception){
                println(e)
            }
            val fileChannel = file.inputStream().channel
            val startOffset = 0
            val declaredLength = file.length()
            println("File opened with length $declaredLength")
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset.toLong(), declaredLength)
        }

        fun box_iou(a: RectF, b: RectF): Float {
            return box_intersection(a, b) / box_union(a, b)
        }

        fun box_intersection(a: RectF, b: RectF): Float {
            val w: Float = overlap(
                (a.left + a.right) / 2, a.right - a.left,
                (b.left + b.right) / 2, b.right - b.left
            )
            val h: Float = overlap(
                (a.top + a.bottom) / 2, a.bottom - a.top,
                (b.top + b.bottom) / 2, b.bottom - b.top
            )
            return if (w < 0.0 || h < 0.0) 0f else (w * h)
        }

        fun box_union(a: RectF, b: RectF): Float {
            val i = box_intersection(a, b)
            return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
        }

        fun overlap(x1: Float, w1: Float, x2: Float, w2: Float): Float {
            val l1 = x1 - w1 / 2
            val l2 = x2 - w2 / 2
            val left = if (l1 > l2) l1 else l2
            val r1 = x1 + w1 / 2
            val r2 = x2 + w2 / 2
            val right = if (r1 < r2) r1 else r2
            return right - left
        }

    }
}