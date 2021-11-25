package com.anlehu.mmhcods.utils

import com.anlehu.mmhcods.utils.Detector.*


class DetectionComparator {

    companion object: Comparator<Detection>{
        override fun compare(lhs: Detection, rhs: Detection): Int{
            return rhs.confidence.compareTo(lhs.confidence)
        }
    }

}