package com.anlehu.mmhcods

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.pm.ActivityInfo
import android.os.Bundle
import android.os.storage.OnObbStateChangeListener
import android.os.storage.StorageManager
import android.util.Log
import android.view.View
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.OpenCVLoader

class MainActivity : AppCompatActivity() {

    private lateinit var startButton: Button
    private lateinit var testButton: Button
    private var mountedPath = ""

    /**
     * On create function of main activity.
     * @param savedInstanceState - Bundle? object
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //Open in Landscape
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        window.decorView.apply{
            systemUiVisibility = View.SYSTEM_UI_FLAG_FULLSCREEN
        }
        setContentView(R.layout.activity_main)

        OpenCVLoader.initDebug()
        mountObb()

        startButton = findViewById(R.id.start_button)
        startButton.setOnClickListener {
            val intent = Intent(this@MainActivity, DetectorActivity::class.java)
            intent.putExtra("path", mountedPath)
            startActivity(intent)
        }

        testButton = findViewById(R.id.test_button)
        testButton.setOnClickListener{
            startActivity(
                Intent(
                    this@MainActivity,
                    TestActivity::class.java
                ).putExtra("mountedPath", mountedPath)
            )
        }

        val activityManager = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        val configurationInfo = activityManager.deviceConfigurationInfo

//        System.err.println(configurationInfo.glEsVersion.toDouble())
//        System.err.println(configurationInfo.reqGlEsVersion >= 0x30000)
//        System.err.println(String.format("%X", configurationInfo.reqGlEsVersion))

    }
    /********************************************************************************************************
     * Mounts OBB
     ********************************************************************************************************/
    private fun mountObb(){
        val storageManager = this.getSystemService(Context.STORAGE_SERVICE) as StorageManager
        val obbPath = this.obbDir.absolutePath
        val mListener = object: OnObbStateChangeListener(){       // OBB state change listener
            override fun onObbStateChange(path: String, state: Int) {
                super.onObbStateChange(path, state)
                Log.d("HELLO_W", "TEST VALUE $state")
                when (state) {
                    MOUNTED -> {
                        Log.d("MOUNTED AT:", storageManager.getMountedObbPath(path))
                        mountedPath = storageManager.getMountedObbPath(path)
                    }
                    ERROR_ALREADY_MOUNTED -> {
                        Log.d("ALREADY MOUNTED:", path)
                    }
                    else -> {
                        Log.d("ERROR:", path)
                    }
                }
            }
        }
        try{
              val y = storageManager!!.mountObb(obbPath+"/main.1.com.anlehu.mmhcods.obb", null, mListener)
        }catch(e: Exception){
            Log.d("ERROR", e.toString())
        }

    }

    /**
     * companion objects
     */
    companion object{

    }
}