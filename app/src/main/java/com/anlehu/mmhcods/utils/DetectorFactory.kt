package com.anlehu.mmhcods.utils

import android.util.Log
import com.anlehu.mmhcods.LaneClassifier
import com.anlehu.mmhcods.YoloV5Classifier

class DetectorFactory {

    companion object{
        fun getDetector(mountedPath: String, modelFileName: String): YoloV5Classifier{

            var labelFileName = ""
            var isQuantized: Boolean = false
            var inputSize = 0

            if (modelFileName == "main_detector_trike.tflite") {
                labelFileName = "main_labels.txt"
                isQuantized = false
                inputSize = 640
            }

            return YoloV5Classifier.create(mountedPath, modelFileName, labelFileName, isQuantized,
                inputSize)
        }

        fun getLaneDetector(mountedPath: String, modelFileName: String): LaneClassifier{

            var isQuantized: Boolean = false
            var inputHeight = 288
            var inputWidth = 800

            if (modelFileName == "lane_detector.tflite") {
                Log.d("INIT", "INITIALIZE LANE DETECTOR")
                isQuantized = false
                inputWidth = 800
                inputHeight = 288
            }

            return LaneClassifier.create(mountedPath, modelFileName, isQuantized,
                inputWidth, inputHeight)
        }
    }

}