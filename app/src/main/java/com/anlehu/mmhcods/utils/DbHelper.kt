package com.anlehu.mmhcods.utils

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import android.provider.BaseColumns

class DbHelper(context: Context): SQLiteOpenHelper(context, DATABASE_NAME, null, DATABASE_VERSION) {

    /********************************************************************************************************
     * Object Initialization
     ********************************************************************************************************/
    object ViolationEntry : BaseColumns{
        const val TABLE_NAME = "violations_table"
        const val COL_TIMESTAMP = "timestamp"
        const val COL_LOC = "location"
        const val COL_SNAPSHOT = "snapshot"
        const val COL_OFFENSE = "offense"
        const val COL_LP = "license_plate"
    }

    /********************************************************************************************************
     * On create method
     ********************************************************************************************************/
    override fun onCreate(db: SQLiteDatabase?) {
        db!!.execSQL(SQL_CREATE_ENTRIES)
    }

    /********************************************************************************************************
     * On upgrade method
     ********************************************************************************************************/
    override fun onUpgrade(db: SQLiteDatabase?, old: Int, new: Int) {
        db!!.execSQL(SQL_DELETE_ENTRIES)
        onCreate(db)
    }

    /********************************************************************************************************
     * On downgrade method
     ********************************************************************************************************/
    override fun onDowngrade(db: SQLiteDatabase?, old: Int, new: Int) {
        onUpgrade(db, old, new)
    }

    /********************************************************************************************************
     * Companion Object
     * Creates SQL Table
     * Deletes specific entries from SQL Table
     ********************************************************************************************************/
    companion object{
        private const val SQL_CREATE_ENTRIES =
            "CREATE TABLE ${ViolationEntry.TABLE_NAME} ("+
                    "${BaseColumns._ID} INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"+
                    "${ViolationEntry.COL_TIMESTAMP} TEXT,"+
                    "${ViolationEntry.COL_LOC} TEXT,"+
                    "${ViolationEntry.COL_SNAPSHOT} TEXT,"+
                    "${ViolationEntry.COL_OFFENSE} TEXT,"+
                    "${ViolationEntry.COL_LP} TEXT)"

        private const val SQL_DELETE_ENTRIES = "DROP TABLE IF EXISTS ${ViolationEntry.TABLE_NAME}"

        const val DATABASE_VERSION = 1
        const val DATABASE_NAME = "Violations.db"


    }
}