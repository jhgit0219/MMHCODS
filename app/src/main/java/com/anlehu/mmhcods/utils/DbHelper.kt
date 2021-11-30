package com.anlehu.mmhcods.utils

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.provider.BaseColumns

class DbHelper(context: Context): SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {


    object violationEntry : BaseColumns{
        const val TABLE_NAME = "violations_table"
        const val COL_TIMESTAMP = "timestamp"
        const val COL_LOC = "location"
        const val COL_SNAPSHOT = "snapshot"
        const val COL_OFFENSE = "offense"
        const val COL_LP = "license_plate"
    }

    private val SQL_CREATE_ENTRIES =
        "CREATE TABLE ${violationEntry.TABLE_NAME} ("+
                "${BaseColumns._ID} INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"+
                "${violationEntry.COL_TIMESTAMP} TEXT,"+
                "${violationEntry.COL_LOC} TEXT,"+
                "${violationEntry.COL_SNAPSHOT} TEXT,"+
                "${violationEntry.COL_OFFENSE} TEXT,"+
                "${violationEntry.COL_LP} TEXT)"

    private val SQL_DELETE_ENTRIES = "DROP TABLE IF EXISTS ${violationEntry.TABLE_NAME}"


    override fun onCreate(db: SQLiteDatabase?) {
        db!!.execSQL(SQL_CREATE_ENTRIES)
    }

    override fun onUpgrade(db: SQLiteDatabase?, old: Int, new: Int) {
        db!!.execSQL(SQL_DELETE_ENTRIES)
        onCreate(db)
    }

    override fun onDowngrade(db: SQLiteDatabase?, old: Int, new: Int) {
        onUpgrade(db, old, new)
    }

    companion object{
        const val DATABASE_VERSION = 1
        const val DATABASE_NAME = "Violations.db"
    }
}