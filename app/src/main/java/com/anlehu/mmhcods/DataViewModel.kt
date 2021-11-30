package com.anlehu.mmhcods

import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import java.util.ArrayList

class DataViewModel: ViewModel() {

    private var tempList : MutableList<DetectorActivity.MotorcycleObject> = ArrayList<DetectorActivity.MotorcycleObject>()
    private var finalTempList : MutableList<DetectorActivity.MotorcycleObject> = ArrayList<DetectorActivity.MotorcycleObject>()
    private var reportedTempList: MutableList<String> = ArrayList<String>()

    val potentialViolationsList: MutableLiveData<MutableList<DetectorActivity.MotorcycleObject>> = MutableLiveData()
    val finalViolationsList: MutableLiveData<MutableList<DetectorActivity.MotorcycleObject>> = MutableLiveData()
    val reportedLicensePlateList: MutableLiveData<MutableList<String>> = MutableLiveData()

    init{
        potentialViolationsList.value = tempList
        finalViolationsList.value = finalTempList
        reportedLicensePlateList.value = reportedTempList
    }

    @Synchronized
    fun addToPotentialViolationsList(motorcycleObject: DetectorActivity.MotorcycleObject){
        tempList = potentialViolationsList.value!!
        tempList.add(motorcycleObject)
        potentialViolationsList.postValue(tempList)
    }
    @RequiresApi(Build.VERSION_CODES.N)
    @Synchronized
    fun removeFromPotentialViolationsList(motorcycleObject: DetectorActivity.MotorcycleObject){
        tempList = potentialViolationsList.value!!
        tempList.removeIf{it == motorcycleObject}
        potentialViolationsList.postValue(tempList)
    }
    @Synchronized
    fun addToFinalViolationsList(motorcycleObject: DetectorActivity.MotorcycleObject){
        finalTempList = finalViolationsList.value!!
        finalTempList.add(motorcycleObject)
        finalViolationsList.postValue(finalTempList)
    }
    @RequiresApi(Build.VERSION_CODES.N)
    @Synchronized
    fun removeFromFinalViolationsList(motorcycleObject: DetectorActivity.MotorcycleObject){
        finalTempList = finalViolationsList.value!!
        finalTempList.removeIf{it == motorcycleObject}
        finalViolationsList.postValue(finalTempList)
    }
    @Synchronized
    fun addToReportedList(ln: String){
        reportedTempList = reportedLicensePlateList.value!!
        reportedTempList.add(ln)
        reportedLicensePlateList.postValue(reportedTempList)
    }
    @RequiresApi(Build.VERSION_CODES.N)
    @Synchronized
    fun removeFromReportedList(ln: String){
        reportedTempList = reportedLicensePlateList.value!!
        reportedTempList.removeIf{it == ln}
        reportedLicensePlateList.postValue(reportedTempList)
    }
    @Synchronized
    fun getPotentialViolationsList(): MutableList<DetectorActivity.MotorcycleObject>? {
        return potentialViolationsList.value
    }
    @Synchronized
    fun getFinalViolationsList(): MutableList<DetectorActivity.MotorcycleObject>? {
        return finalViolationsList.value
    }
    @Synchronized
    fun getReportedList(): MutableList<String>? {
        return reportedLicensePlateList.value
    }
    @Synchronized
    fun setPotentialViolationsList(list: MutableList<DetectorActivity.MotorcycleObject>){
        potentialViolationsList.postValue(list)
    }
    @Synchronized
    fun setFinalViolationsList(list: MutableList<DetectorActivity.MotorcycleObject>){
        potentialViolationsList.postValue(list)
    }
    @Synchronized
    fun setReportedList(list: MutableList<String>){
        reportedLicensePlateList.postValue(list)
    }


}