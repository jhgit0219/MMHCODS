package com.anlehu.mmhcods.utils

import android.content.res.AssetManager
import java.io.File
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class Utils {
    companion object{
        @Throws(IOException::class)
        fun loadModelFile(assetManager: AssetManager, modelFilename: String?): MappedByteBuffer? {
//            lateinit var fileDescriptor: AssetFileDescriptor
//            try{
//                 fileDescriptor = assetManager.openNonAssetFd("$modelFilename")
//            }catch (e: Exception){
//                println(e.toString())
//            }
            lateinit var file: File
            try{
                file = File(modelFilename!!)
            }catch(e: Exception){
                println(e)
            }

            //val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            //val fileChannel = inputStream.channel
            val fileChannel = file.inputStream().channel
            val startOffset = 0
            val declaredLength = file.length()
            println("File opened with length $declaredLength")
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset.toLong(), declaredLength)
        }

    }
}