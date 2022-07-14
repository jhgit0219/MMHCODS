package com.anlehu.mmhcods.utils

import android.content.res.AssetManager
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ModelUtils {
    /********************************************************************************************************
     * Companion Object
     * Loading model file for detection
     ********************************************************************************************************/
    companion object{
        /********************************************************************************************************
         * Function that loads model to be used for detection
         ********************************************************************************************************/
        @Throws(IOException::class)
        fun loadModelFile(assets: AssetManager, modelFilename: String?): MappedByteBuffer? {
            val fileDescriptor = assets.openFd(modelFilename!!)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }

    }
}