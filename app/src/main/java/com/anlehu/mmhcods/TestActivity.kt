package com.anlehu.mmhcods

import android.app.Activity
import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.provider.DocumentsContract
import android.util.Log
import android.util.Size
import android.widget.Button
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.anlehu.mmhcods.utils.*
import com.anlehu.mmhcods.views.OverlayView
import java.io.File
import java.util.*

/**
 * Performs violation testing for research/graphing purposes
 */

class TestActivity : AppCompatActivity() {

    /********************************************************************************************************
     * Variable Initializaitons
     ********************************************************************************************************/

    private var handlerThread: HandlerThread? = null
    private var handler: Handler? = null

    private lateinit var dirButton: Button
    private lateinit var trackingOverlay: OverlayView
    private lateinit var frameToCropMat: Matrix
    private lateinit var cropToFrameMat: Matrix
    private lateinit var frameToLaneMat: Matrix
    private lateinit var laneToFrameMat: Matrix
    private lateinit var borderedText: BorderedText
    private lateinit var tracker: BoxTracker
    private lateinit var detector: YoloV5Classifier
    private lateinit var laneDetector: LaneClassifier

    private lateinit var finViolationObserver: androidx.lifecycle.Observer<MutableList<DetectorActivity.MotorcycleObject>>

    private var sensorOrientation: Int = 0
    private var previewWidth = 0
    private var previewHeight = 0
    private var computingDetection: Boolean = false


    private var croppedBitmap: Bitmap? = null
    private var laneBitmap: Bitmap? = null
    private var copiedBitmap: Bitmap? = null

    private var DESIRED_PREVIEW_SIZE = Size(1920, 1080)

    private var MAINTAIN_ASPECT = true

    /********************************************************************************************************
     * Starts activity for result
     ********************************************************************************************************/
    @RequiresApi(Build.VERSION_CODES.N)
    var resultStarter = registerForActivityResult(ActivityResultContracts.StartActivityForResult()){ result ->
        if(result.resultCode == Activity.RESULT_OK){
            val data: Intent? = result.data
            if (data != null) {
                gatherPhotos(data)
            }
        }
    }

    /********************************************************************************************************
     * On create method
     ********************************************************************************************************/
    override fun onCreate(savedInstanceState: Bundle?){
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_test)

        initializeDetectors()

        dirButton = findViewById(R.id.dir_button)
        dirButton.setOnClickListener{
            var intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
            intent.addCategory(Intent.CATEGORY_DEFAULT)
            resultStarter.launch(Intent.createChooser(intent, "Choose Directory"))
        }

    }

    /********************************************************************************************************
     * Retrieves all photos within selected directory
     ********************************************************************************************************/
    @RequiresApi(Build.VERSION_CODES.N)
    fun gatherPhotos(data: Intent){

        /*
            Get photo dir.
            Pop up to select folder containing images.
         */

        var uri: Uri? = data.data
        var docUri: Uri = DocumentsContract.buildDocumentUriUsingTree(uri,
            DocumentsContract.getTreeDocumentId(uri))
        var path = FileUtil.getFullPathFromTreeUri(uri, this)
        Log.d("PROC_PHOTO", "Directory: " +path.toString())

        var directory = File(path)
        var files: Array<File> = directory.listFiles()
        Log.d("FILES_SIZE", "Size: ${files.size}")

        // Check if location contains jpeg


        for(item in files){
            if(item.toString().endsWith(".jpg") or item.toString().endsWith(".png")
                or item.toString().endsWith(".jpeg")){
                Log.d("FILE_ITEM", "$item")

                croppedBitmap = ImageUtils.rescaleImage(null, item.toString())
                /*
                    Apply testing per image
                */
                processImage()
            }
        }


    }

    /********************************************************************************************************
     * Processes current image
     ********************************************************************************************************/
    @RequiresApi(Build.VERSION_CODES.N)
    fun processImage(){
        // run in a background thread
        //runInBackground {

            val results: List<Detector.Detection> = detector.detectImage(croppedBitmap!!)
            val lane  = laneDetector.detectLane(laneBitmap!!)
            copiedBitmap = Bitmap.createBitmap(croppedBitmap!!)
            val paint = Paint()
            paint.color = Color.BLUE
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.0f

            val minConfidence = MainActivity.MINIMUM_CONFIDENCE // sets minimum confidence levels
                                                                // for detection

            // create a list for detection map
            var mappedPredictions: MutableList<Detector.Detection> = LinkedList()

            /*
                Go through every result of the object detection
             */
            for (result in results) {
                val location: RectF = result.location           // get RectF location of detected object in image if any

                Log.d("DETECTION",result.detectedClass.toString())

                // if location is not null and the detection confidence is over threshold, then...
                if (location != null && result.confidence >= minConfidence) {
                    //canvas1.drawRect(location, paint)           // draw on canvas1 for the location of detected object
                    cropToFrameMat.mapRect(location)            // add location rects to cropToFrameMat
                    //result.location = location
                    mappedPredictions.add(result)               // add result to mappedPredictions
                }
            }

            // Detect Violations
            DetectorActivity.detectViolations(mappedPredictions)

            computingDetection = false
        //}
    }

    /********************************************************************************************************
     * Initialize all the detectors
     ********************************************************************************************************/
    fun initializeDetectors(){
        try{
            detector = DetectorFactory.getDetector(assets, MainActivity.MAIN_MODEL_NAME)
            laneDetector = DetectorFactory.getLaneDetector(assets, MainActivity.LANE_MODEL_NAME)
        }catch(e: Exception){
            Toast.makeText(this, "Classifier/s can't be initiated", Toast.LENGTH_SHORT).show()
            finish()
        }

        val cropSize = detector.getInputSize()
        val inputWidth = laneDetector.INPUT_WIDTH
        val inputHeight = laneDetector.INPUT_HEIGHT

        setPreviewDimensions(1920, 1080)
        DetectorActivity.rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        laneBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)

        frameToCropMat = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight, cropSize, cropSize, sensorOrientation, MAINTAIN_ASPECT
        )
        // lane matrix
        frameToLaneMat = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight, inputWidth, inputHeight, sensorOrientation, MAINTAIN_ASPECT
        )

        cropToFrameMat = Matrix()
        frameToCropMat.invert(cropToFrameMat)

        laneToFrameMat = Matrix()
        frameToLaneMat.invert(laneToFrameMat)
    }

    /********************************************************************************************************
     * Sets preview dimensions
     ********************************************************************************************************/
    fun setPreviewDimensions(width: Int, height: Int){
        previewWidth = width
        previewHeight = height
    }

    /********************************************************************************************************
     * Actions needed when process is resumed
     ********************************************************************************************************/
    @Synchronized
    override fun onResume(){
        super.onResume()
        handlerThread = HandlerThread("inference")
        handlerThread!!.start()
        handler = Handler(handlerThread!!.looper)
    }

    /********************************************************************************************************
     * Actions needed when process is paused
     ********************************************************************************************************/
    @Synchronized
    override fun onPause() {
        handlerThread!!.quitSafely()
        try {
            handlerThread!!.join()
            handlerThread = null
            handler = null
        } catch (e: InterruptedException) {
            Log.d("onPause", e.toString())
        }
        super.onPause()
    }

    /********************************************************************************************************
     * Action performed when process is stopped
     ********************************************************************************************************/
    @Synchronized
    override fun onStop() {
        super.onStop()
    }

    /********************************************************************************************************
     * Action performed when process is destroyed
     ********************************************************************************************************/
    @Synchronized
    override fun onDestroy() {
        super.onDestroy()
    }

}