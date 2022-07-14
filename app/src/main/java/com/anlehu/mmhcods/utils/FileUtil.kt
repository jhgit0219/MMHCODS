package com.anlehu.mmhcods.utils

import android.annotation.TargetApi
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.storage.StorageManager
import android.provider.DocumentsContract
import androidx.annotation.RequiresApi
import java.io.File
import java.lang.reflect.Method
import java.time.Duration
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.TimeUnit


class FileUtil {
    /********************************************************************************************************
     * Companion Object
     * Gets and sets Date Time
     * Gets different Full File Paths, Document Paths, and Volume Paths
     ********************************************************************************************************/
    @RequiresApi(Build.VERSION_CODES.O)
    companion object {

        /********************************************************************************************************
         * Variable Initializaitons
         ********************************************************************************************************/
        private val PRIMARY_VOLUME_NAME = "primary"

        private val format = DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss")

        /********************************************************************************************************
         * Gets current date time
         * @return value of current local date time
         ********************************************************************************************************/
        fun getDateTime(): LocalDateTime {
            return LocalDateTime.now()
        }

        /********************************************************************************************************
         * Function that uses Date input in String form to Local Date Time value
         * @return string input converted into local date time value
         ********************************************************************************************************/
        fun stringToDate(str: String): LocalDateTime {
            return LocalDateTime.parse(str, format)
        }

        /********************************************************************************************************
         * Function that uses LocalDateTime input to return a string value of that input
         * @return local date time ni string format
         ********************************************************************************************************/
        fun DateToString(time: LocalDateTime): String{
            return time.format(format)
        }


        /********************************************************************************************************
         * Function that calculates duration between Time 1 and Time 2
         * @return duration in milliseconds between 2 different times
         ********************************************************************************************************/
        fun elapsedTime(time_1: LocalDateTime, time_2: LocalDateTime): Long {
            var duration = Duration.between(time_1, time_2).toMillis()
            return TimeUnit.MILLISECONDS.toSeconds(duration)
        }

        /********************************************************************************************************
         * Gets full file path from tree URI
         * @return full file path value
         ********************************************************************************************************/
        fun getFullPathFromTreeUri(treeUri: Uri?, con: Context?): String? {
            if (treeUri == null) return null
            var volumePath = getVolumePath(getVolumeIdFromTreeUri(treeUri), con!!) ?: return File.separator
            if (volumePath.endsWith(File.separator)) volumePath = volumePath.substring(0, volumePath.length - 1)
            var documentPath = getDocumentPathFromTreeUri(treeUri)
            if (documentPath != null) {
                if (documentPath.endsWith(File.separator)) documentPath =
                    documentPath!!.substring(0, documentPath.length - 1)
            }
            return if (documentPath!!.isNotEmpty()) {
                if (documentPath.startsWith(File.separator)) volumePath + documentPath else volumePath + File.separator.toString() + documentPath
            } else volumePath
        }

        /********************************************************************************************************
         * Gets volume path
         * @return volume path value
         ********************************************************************************************************/
        private fun getVolumePath(volumeId: String?, context: Context): String? {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) return null
            return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) getVolumePathForAndroid11AndAbove(
                volumeId,
                context
            ) else getVolumePathBeforeAndroid11(volumeId, context).toString()
        }

        /********************************************************************************************************
         * Gets volume path on systems before Android11
         * @return volume path value
         ********************************************************************************************************/
        private fun getVolumePathBeforeAndroid11(volumeId: String?, context: Context): Any? {
            return try {
                val mStorageManager = context.getSystemService(Context.STORAGE_SERVICE) as StorageManager
                val storageVolumeClazz = Class.forName("android.os.storage.StorageVolume")
                val getVolumeList: Method = mStorageManager.javaClass.getMethod("getVolumeList")
                val getUuid: Method = storageVolumeClazz.getMethod("getUuid")
                val getPath: Method = storageVolumeClazz.getMethod("getPath")
                val isPrimary: Method = storageVolumeClazz.getMethod("isPrimary")
                val result: Any = getVolumeList.invoke(mStorageManager)
                val length: Int = (result as Array<*>).size
                for (i in 0 until length) {
                    val storageVolumeElement: Any? = result[i]
                    val uuid = getUuid.invoke(storageVolumeElement) as String
                    val primary = isPrimary.invoke(storageVolumeElement) as Boolean
                    if (primary && PRIMARY_VOLUME_NAME == volumeId) // primary volume?
                        return getPath.invoke(storageVolumeElement).toString()
                    if (uuid != null && uuid == volumeId) // other volumes?
                        return getPath.invoke(storageVolumeElement)
                }
                // not found.
                null
            } catch (ex: Exception) {
                null
            }
        }

        /********************************************************************************************************
         * Gets volume path on systems post Android11
         * @return volume path value
         ********************************************************************************************************/
        @TargetApi(Build.VERSION_CODES.R)
        private fun getVolumePathForAndroid11AndAbove(volumeId: String?, context: Context): String? {
            return try {
                val mStorageManager = context.getSystemService(Context.STORAGE_SERVICE) as StorageManager
                val storageVolumes = mStorageManager.storageVolumes
                for (storageVolume in storageVolumes) {
                    // primary volume?
                    if (storageVolume.isPrimary && PRIMARY_VOLUME_NAME == volumeId) return storageVolume.directory!!
                        .path

                    // other volumes?
                    val uuid = storageVolume.uuid
                    if (uuid != null && uuid == volumeId) return storageVolume.directory!!.path
                }
                // not found.
                null
            } catch (ex: Exception) {
                null
            }
        }

        /********************************************************************************************************
         * Gets volume path from tree URI
         * @return volume path
         ********************************************************************************************************/
        @TargetApi(Build.VERSION_CODES.LOLLIPOP)
        private fun getVolumeIdFromTreeUri(treeUri: Uri): String? {
            val docId = DocumentsContract.getTreeDocumentId(treeUri)
            val split = docId.split(":".toRegex()).toTypedArray()
            return if (split.size > 0) split[0] else null
        }

        /********************************************************************************************************
         * Gets document path from tree URI
         * @return document path
         ********************************************************************************************************/
        @TargetApi(Build.VERSION_CODES.LOLLIPOP)
        private fun getDocumentPathFromTreeUri(treeUri: Uri): String? {
            val docId = DocumentsContract.getTreeDocumentId(treeUri)
            val split: Array<String?> = docId.split(":".toRegex()).toTypedArray()
            return if (split.size >= 2 && split[1] != null) split[1] else File.separator
        }
    }


}