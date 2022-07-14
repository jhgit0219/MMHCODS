package com.anlehu.mmhcods.utils

import android.graphics.Bitmap
import android.graphics.RectF

interface Detector {

    /********************************************************************************************************
     * Function Initializations
     ********************************************************************************************************/
    fun detectImage(bitmap: Bitmap): ArrayList<Detection>

    fun close()

    fun setNumOfThreads(numThread: Int)

    fun setUseNNAPI(isChecked: Boolean)

    fun getObjThresh(): Float

    /********************************************************************************************************
     * result returned by detector
     ********************************************************************************************************/
    open class Detection{
        /********************************************************************************************************
         * Unique ID for what has been detected
         ********************************************************************************************************/
        var id: String

        /********************************************************************************************************
         * Name for detection
         ********************************************************************************************************/
        var title: String

        /********************************************************************************************************
         * Value that shows how confident the model is about the detected class in the frame. Higher is better.
         ********************************************************************************************************/
        var confidence: Float = 0f

        /********************************************************************************************************
         * Location of the detected object within the frame
         ********************************************************************************************************/

        /********************************************************************************************************
         * Variable Initializations
         ********************************************************************************************************/
        var location: RectF

        var detectedClass: Int = -1

        /********************************************************************************************************
         * Class constructor
         ********************************************************************************************************/
        constructor(): super(){
            id = ""
            title = ""
            confidence = 0f
            location = RectF()
        }

        /********************************************************************************************************
         * Class constructor copying detection object attributes
         ********************************************************************************************************/
        constructor(detection: Detection): super(){
            this.id = detection.id
            this.title = detection.title
            this.confidence = detection.confidence
            this.location = detection.location
        }

        /********************************************************************************************************
         * Class constructor
         ********************************************************************************************************/
        constructor(
            id: String,
            title: String,
            confidence: Float,
            location: RectF
        ): super(){
            this.id = id
            this.title = title
            this.confidence = confidence
            this.location = location
        }

        /********************************************************************************************************
         * Class constructor
         ********************************************************************************************************/
        constructor(
            id: String,
            title: String,
            confidence: Float,
            location: RectF,
            detectedClass: Int
        ):super(){
            this.id = id
            this.title = title
            this.confidence = confidence
            this.location = location
            this.detectedClass = detectedClass
        }

        /********************************************************************************************************
         * Converting class characteristics into a single string
         * @return a single string containing combined class characteristics
         ********************************************************************************************************/
        override fun toString(): String{

            var resultString: String = ""
            if(id != null){
                resultString += "[$id]"
            }
            if(title != null){
                resultString += "[$title]"
            }
            if(confidence != null){
                resultString += String.format("(%.1f%%) ", confidence * 100.0f)
            }
            if(location != null){
                resultString += location.toString()
            }
            return resultString.trim()
        }

    }



}
