package com.anlehu.mmhcods.utils

import android.content.res.AssetManager
import android.util.Log
import com.anlehu.mmhcods.LaneClassifier
import com.anlehu.mmhcods.YoloV5Classifier

class DetectorFactory {

    companion object{
        fun getDetector(assetManager: AssetManager, modelFileName: String): YoloV5Classifier{
            var labelFileName = ""
            var isQuantized: Boolean = false
            var inputSize = 0

            if (modelFileName == "main_detector_trike.tflite") {
                labelFileName = "file:///android_asset/main_labels.txt"
                isQuantized = false
                inputSize = 640
            }

            return YoloV5Classifier.create(assetManager, modelFileName, labelFileName, isQuantized,
                inputSize)
        }

        /*fun getLaneDetector(assetManager: AssetManager, modelFileName: String): LaneClassifier{
            var labelFileName = ""
            var isQuantized: Boolean = false
            var inputHeight = 288
            var inputWidth = 600

            if (modelFileName == "lane_detector.tflite") {
                Log.d("INIT", "INITIALIZE LANE DETECTOR")
                //labelFileName = "file:///android_asset/lane_labels.txt"
                isQuantized = false
                inputWidth = 288
                inputHeight = 600
            }

           /* return LaneClassifier.create(assetManager, modelFileName, labelFileName, isQuantized,
                inputWidth, inputHeight)*/
        }*/
    }

}