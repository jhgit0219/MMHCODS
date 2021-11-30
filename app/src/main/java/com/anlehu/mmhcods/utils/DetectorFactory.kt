package com.anlehu.mmhcods.utils

import android.content.res.AssetManager
import com.anlehu.mmhcods.YoloV5Classifier

class DetectorFactory {

    companion object{
        fun getDetector(assetManager: AssetManager, modelFileName: String): YoloV5Classifier{
            var labelFileName = ""
            var isQuantized: Boolean = false
            var inputSize = 0

            if (modelFileName == "main_detector.tflite") {
                labelFileName = "file:///android_asset/main_labels.txt"
                isQuantized = false
                inputSize = 640
            }

            return YoloV5Classifier.create(assetManager, modelFileName, labelFileName, isQuantized,
                inputSize)
        }
    }

}