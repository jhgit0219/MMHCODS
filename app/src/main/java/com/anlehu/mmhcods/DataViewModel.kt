package com.anlehu.mmhcods

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import java.util.*
import java.util.concurrent.Semaphore

class DataViewModel: ViewModel() {

    /********************************************************************************************************
     * Variable Initializations
     ********************************************************************************************************/

    private var tempList : MutableList<DetectorActivity.MotorcycleObject> = ArrayList<DetectorActivity.MotorcycleObject>()
    private var finalTempList : MutableList<DetectorActivity.MotorcycleObject> = ArrayList<DetectorActivity.MotorcycleObject>()
    private var reportedTempList: MutableList<String> = ArrayList<String>()
    private val potLock = Semaphore(1)
    private val finLock = Semaphore(1)
    private val repLock = Semaphore(1)

    val potentialViolationsList: MutableLiveData<MutableList<DetectorActivity.MotorcycleObject>> = MutableLiveData()
    val finalViolationsList: MutableLiveData<MutableList<DetectorActivity.MotorcycleObject>> = MutableLiveData()
    val reportedLicensePlateList: MutableLiveData<MutableList<String>> = MutableLiveData()

    init{
        potentialViolationsList.value = tempList
        finalViolationsList.value = finalTempList
        reportedLicensePlateList.value = reportedTempList
    }

    /********************************************************************************************************
     * Function that adds current motorcycleObject to potentialViolationList
     ********************************************************************************************************/
    @Synchronized
    fun addToPotentialViolationsList(motorcycleObject: DetectorActivity.MotorcycleObject){
        try{
            potLock.acquire()
            tempList = potentialViolationsList.value!!
            tempList.add(motorcycleObject)
            potentialViolationsList.postValue(tempList)
        }catch(e: Exception){
            e.printStackTrace()
        }finally{
            potLock.release()
        }

    }
    /********************************************************************************************************
     * Function that removes a motorcycleObject from potentialViolationList
     ********************************************************************************************************/
    @RequiresApi(Build.VERSION_CODES.N)
    @Synchronized
    fun removeFromPotentialViolationsList(motorcycleObject: DetectorActivity.MotorcycleObject){
        try{
            potLock.acquire()
            tempList = potentialViolationsList.value!!
            tempList.removeIf{it == motorcycleObject}
            potentialViolationsList.postValue(tempList)
        }catch(e: Exception){
            e.printStackTrace()
        }finally{
            potLock.release()
        }
    }
    /********************************************************************************************************
     * Function that adds motorcycleObject to finalViolationList
     ********************************************************************************************************/
    @Synchronized
    fun addToFinalViolationsList(motorcycleObject: DetectorActivity.MotorcycleObject){
        try{
            finLock.acquire()
            finalTempList = finalViolationsList.value!!
            finalTempList.add(motorcycleObject)
            finalViolationsList.postValue(finalTempList)
        }catch(e: Exception){
            e.printStackTrace()
        }finally{
            finLock.release()
        }
    }
    /********************************************************************************************************
     * Function that removes a motorcycleObject from finalViolationList
     ********************************************************************************************************/
    @RequiresApi(Build.VERSION_CODES.N)
    @Synchronized
    fun removeFromFinalViolationsList(motorcycleObject: DetectorActivity.MotorcycleObject){
        try{
            finLock.acquire()
            finalTempList = finalViolationsList.value!!
            finalTempList.removeIf{it == motorcycleObject}
            finalViolationsList.postValue(finalTempList)
        }catch(e: Exception){
            e.printStackTrace()
        }finally{
            finLock.release()
        }

    }
    /********************************************************************************************************
     * Adds motorcycleObject's license plate value to reportedList
     ********************************************************************************************************/
    @Synchronized
    fun addToReportedList(ln: String){
        try{
            repLock.acquire()
            reportedTempList = reportedLicensePlateList.value!!
            reportedTempList.add(ln)
            reportedLicensePlateList.postValue(reportedTempList)
        }catch(e: Exception){
            e.printStackTrace()
        }finally{
            repLock.release()
        }
    }
    /********************************************************************************************************
     * Removes motorcycleObject's license plate value from reportedList
     ********************************************************************************************************/
    @RequiresApi(Build.VERSION_CODES.N)
    @Synchronized
    fun removeFromReportedList(ln: String){
        try{
            repLock.acquire()
            reportedTempList = reportedLicensePlateList.value!!
            reportedTempList.removeIf{it == ln}
            reportedLicensePlateList.postValue(reportedTempList)
        }catch(e: Exception){
            e.printStackTrace()
        }finally {
            repLock.release()
        }

    }
    /********************************************************************************************************
     * Gets list of objects from potentialViolationsList
     * @return current list of objects in potentialViolationsList
     ********************************************************************************************************/
    @Synchronized
    fun getPotentialViolationsList(): MutableList<DetectorActivity.MotorcycleObject>? {
        var list: MutableList<DetectorActivity.MotorcycleObject> = ArrayList()
        try{
            potLock.acquire()
             list = potentialViolationsList.value!!
        }catch(e: Exception){
            e.printStackTrace()
        }finally{
            potLock.release()
        }
        return list

    }
    /********************************************************************************************************
     * Gets list of objects from finalViolationsList
     * @return current list of objects in finalViolationsList
     ********************************************************************************************************/
    @Synchronized
    fun getFinalViolationsList(): MutableList<DetectorActivity.MotorcycleObject>? {
        var list: MutableList<DetectorActivity.MotorcycleObject> = ArrayList()
        try{
            finLock.acquire()
            list = finalViolationsList.value!!
        }catch(e: Exception){
            e.printStackTrace()
        }finally{
            finLock.release()
        }
        return list
    }
    /********************************************************************************************************
     * Gets list of values from reportedList
     * @return values from reportedList
     ********************************************************************************************************/
    @Synchronized
    fun getReportedList(): MutableList<String> {
        var list: MutableList<String> = ArrayList()
        try{
            repLock.acquire()
            list = reportedLicensePlateList.value!!
        }catch(e: Exception){
            e.printStackTrace()
        }finally{
            repLock.release()
        }
        return list
    }
    /********************************************************************************************************
     * Sets and populates potentialViolationsList
     ********************************************************************************************************/
    @Synchronized
    fun setPotentialViolationsList(list: MutableList<DetectorActivity.MotorcycleObject>){
        try{
            potLock.acquire()
            potentialViolationsList.postValue(list)
        }catch(e: Exception){
            e.printStackTrace()
        }finally{
            potLock.release()
        }

    }
    /********************************************************************************************************
     * Sets and populates finalViolationsList
     ********************************************************************************************************/
    @Synchronized
    fun setFinalViolationsList(list: MutableList<DetectorActivity.MotorcycleObject>){
        try{
            finLock.acquire()
            finalViolationsList.postValue(list)
        }catch(e: Exception){
            e.printStackTrace()
        }finally{
            finLock.release()
        }

    }
    /********************************************************************************************************
     * Sets and populates reportedList
     ********************************************************************************************************/
    @Synchronized
    fun setReportedList(list: MutableList<String>){
        try{
            repLock.acquire()
            reportedLicensePlateList.postValue(list)
        }catch(e: Exception){
            e.printStackTrace()
        }finally{
            repLock.release()
        }

    }


}