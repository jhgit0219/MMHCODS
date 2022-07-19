package com.anlehu.mmhcods

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.storage.OnObbStateChangeListener
import android.os.storage.StorageManager
import android.util.Log
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

        //initBox()

        val activityManager = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        val configurationInfo = activityManager.deviceConfigurationInfo

        System.err.println(configurationInfo.glEsVersion.toDouble())
        System.err.println(configurationInfo.reqGlEsVersion >= 0x30000)
        System.err.println(String.format("%X", configurationInfo.reqGlEsVersion))

    }

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
//            // Debug LOGS
//            val x = storageManager!!.unmountObb(obbPath+"/main.1.com.anlehu.mmhcods.obb", true, mListener)
//            Log.d("UNMOUNT: ", "$x")
              val y = storageManager!!.mountObb(obbPath+"/main.1.com.anlehu.mmhcods.obb", null, mListener)
//            Log.d("MOUNT: ", "$y at $obbPath")

        }catch(e: Exception){
            Log.d("ERROR", e.toString())
        }

    }

    private fun initBox() {
        /*previewHeight = INPUT_SIZE
        previewWidth = INPUT_SIZE

        frameToCropMatrix = ImageUtils.getTransformationMatrix(
            previewWidth,
            previewHeight,
            INPUT_SIZE,
            INPUT_SIZE,
            sensorOrientation,
            MAINTAIN_ASPECT
        )

        cropToFrameMatrix = Matrix()
        frameToCropMatrix.invert(cropToFrameMatrix)
        tracker = MultiBoxTracker(this)
        overlayView = findViewById(R.id.tracking_overlay)
        overlayView.addCallback { canvas -> tracker.draw(canvas) }
        tracker.setFrameConfiguration(
            MainActivity.TF_OD_API_INPUT_SIZE,
            MainActivity.TF_OD_API_INPUT_SIZE,
            sensorOrientation
        )*/
        /*try {
            detector = YoloV5Classifier.create(
                assets,
                MAIN_MODEL_NAME,
                MAIN_LABELS_NAME,
                IS_QUANTIZED,
                INPUT_SIZE
            )
        } catch (e: IOException) {
            e.printStackTrace()
            val toast = Toast.makeText(
                applicationContext, "Classifier could not be initialized", Toast.LENGTH_SHORT
            )
            toast.show()
            finish()
        }*/
    }

    /**
     * companion objects
     */
    companion object{

        var MINIMUM_CONFIDENCE = 0.4f
        var IS_QUANTIZED = false
        var MAIN_MODEL_NAME: String = "main_detector_trike.tflite"
        var LANE_MODEL_NAME: String = "lane_detector.tflite"
        var MAIN_LABELS_NAME: String = "main_labels.txt"
        var LANE_LABELS_NAME: String = "lane_labels.txt"
        var MAINTAIN_ASPECT = true

    }
}