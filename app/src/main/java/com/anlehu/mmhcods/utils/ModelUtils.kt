package com.anlehu.mmhcods.utils

import java.io.File
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ModelUtils {
    companion object{
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

    }
}