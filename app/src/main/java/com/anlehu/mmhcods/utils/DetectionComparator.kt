package com.anlehu.mmhcods.utils

import com.anlehu.mmhcods.utils.Detector.*


class DetectionComparator {
    /********************************************************************************************************
     * Companion Object
     * Comparison between confidence of two detecitons
     ********************************************************************************************************/
    companion object: Comparator<Detection>{
        /********************************************************************************************************
         * Compares confidence between two detections
         * @return value of difference between confidence of two detections
         ********************************************************************************************************/
        override fun compare(lhs: Detection, rhs: Detection): Int{
            return rhs.confidence.compareTo(lhs.confidence)
        }
    }

}