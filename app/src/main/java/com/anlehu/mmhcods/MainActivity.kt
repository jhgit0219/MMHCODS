package com.anlehu.mmhcods

import android.app.ActivityManager
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.Toast
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var detector: YoloV5Classifier
    private lateinit var startButton: Button


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        startButton = findViewById(R.id.start_button)
        startButton.setOnClickListener {
            startActivity(
                Intent(
                    this@MainActivity,
                    DetectorActivity::class.java
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


    companion object{
        var MINIMUM_CONFIDENCE = 0.4f
        var INPUT_SIZE = 640
        var IS_QUANTIZED = false
        var MAIN_MODEL_NAME: String = "best-fp16.tflite"
        var MAIN_LABELS_NAME: String = "file:///android_asset/best-fp16.txt"

        var MAINTAIN_ASPECT = true

    }
}