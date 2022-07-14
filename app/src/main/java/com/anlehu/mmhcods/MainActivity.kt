package com.anlehu.mmhcods

import android.app.ActivityManager
import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.OpenCVLoader

class MainActivity : AppCompatActivity() {

    /********************************************************************************************************
     * Variable Initialization
     ********************************************************************************************************/

    private lateinit var startButton: Button
    private lateinit var testButton: Button

    /**
     * On create function of main activity.
     * @param savedInstanceState - Bundle? object
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        OpenCVLoader.initDebug()

        startButton = findViewById(R.id.start_button)
        startButton.setOnClickListener {
            startActivity(
                Intent(
                    this@MainActivity,
                    DetectorActivity::class.java
                )
            )
        }

        testButton = findViewById(R.id.test_button)
        testButton.setOnClickListener{
            startActivity(
                Intent(
                    this@MainActivity,
                    TestActivity::class.java
                )
            )
        }

        //initBox()

        val activityManager = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        val configurationInfo = activityManager.deviceConfigurationInfo

        System.err.println(configurationInfo.glEsVersion.toDouble())
        System.err.println(configurationInfo.reqGlEsVersion >= 0x30000)
        System.err.println(String.format("%X", configurationInfo.reqGlEsVersion))

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

    /********************************************************************************************************
     * Companion Object
     * Sets values for Detector related variables
     ********************************************************************************************************/
    companion object{

        var MINIMUM_CONFIDENCE = 0.4f
        var IS_QUANTIZED = false
        var MAIN_MODEL_NAME: String = "main_detector_trike.tflite"
        var LANE_MODEL_NAME: String = "lane_detector.tflite"
        var MAIN_LABELS_NAME: String = "file:///android_asset/main_labels.txt"
        var LANE_LABELS_NAME: String = "file:///android_asset/lane_labels.txt"
        var MAINTAIN_ASPECT = true

    }
}