package com.anlehu.mtrvds

import android.app.ActivityManager
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.Toast
import com.anlehu.mtrvds.utils.Detector
import com.anlehu.mtrvds.utils.ImageUtils
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private var sensorOrientation: Int = 90
    private lateinit var modelDetector: Detector

    private lateinit var frameToCropMatrix: Matrix
    private lateinit var cropToFrameMatrix: Matrix

    private lateinit var tracker: MultiBoxTracker
    private lateinit var overlayView: OverlayView

    private lateinit var sourceBitmap: Bitmap
    private lateinit var cropBitmap: Bitmap

    private lateinit var startButton: Button

    var previewWidth    = 0
    var previewHeight   = 0




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

        initBox()

        val activityManager = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        val configurationInfo = activityManager.deviceConfigurationInfo

        System.err.println(configurationInfo.glEsVersion.toDouble())
        System.err.println(configurationInfo.reqGlEsVersion >= 0x30000)
        System.err.println(String.format("%X", configurationInfo.reqGlEsVersion))

    }

    private fun initBox() {
        previewHeight = INPUT_SIZE
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
        trackingOverlay = findViewById(R.id.tracking_overlay)
        trackingOverlay.addCallback { canvas -> tracker.draw(canvas) }
        tracker.setFrameConfiguration(
            MainActivity.TF_OD_API_INPUT_SIZE,
            MainActivity.TF_OD_API_INPUT_SIZE,
            sensorOrientation
        )
        try {
            detector = YoloV5Classifier.create(
                assets,
                MainActivity.TF_OD_API_MODEL_FILE,
                MainActivity.TF_OD_API_LABELS_FILE,
                MainActivity.TF_OD_API_IS_QUANTIZED,
                MainActivity.TF_OD_API_INPUT_SIZE
            )
        } catch (e: IOException) {
            e.printStackTrace()
            MainActivity.LOGGER.e(e, "Exception initializing classifier!")
            val toast = Toast.makeText(
                applicationContext, "Classifier could not be initialized", Toast.LENGTH_SHORT
            )
            toast.show()
            finish()
        }
    }

    companion object{
        var MINIMUM_CONFIDENCE = 0.3f
        var INPUT_SIZE = 320
        var IS_QUANTIZED = false
        var MODEL_NAME: String = "best-fp16.tflite"
        var LABELS_NAME: String = "file:///android_asset/best-fp16.txt"

        var MAINTAIN_ASPECT = true

    }
}