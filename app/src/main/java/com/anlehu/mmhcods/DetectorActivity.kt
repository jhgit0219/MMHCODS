package com.anlehu.mmhcods

import android.content.ContentValues
import android.graphics.*
import android.media.ImageReader
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.util.TypedValue
import android.view.Surface
import android.widget.Toast
import androidx.annotation.RequiresApi
import com.anlehu.mmhcods.utils.BorderedText
import com.anlehu.mmhcods.utils.DbHelper
import com.anlehu.mmhcods.utils.DbHelper.ViolationEntry
import com.anlehu.mmhcods.utils.Detector.Detection
import com.anlehu.mmhcods.utils.DetectorFactory
import com.anlehu.mmhcods.utils.ImageUtils
import com.anlehu.mmhcods.views.OverlayView
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.util.*
import kotlin.collections.ArrayList

class DetectorActivity: CameraActivity(), ImageReader.OnImageAvailableListener {

    private lateinit var trackingOverlay: OverlayView
    private lateinit var frameToCropMat: Matrix
    private lateinit var cropToFrameMat: Matrix
    private lateinit var borderedText: BorderedText
    private lateinit var tracker: BoxTracker
    private lateinit var detector: YoloV5Classifier

    private val liveModel = DataViewModel()
    private lateinit var finViolationObserver: androidx.lifecycle.Observer<MutableList<MotorcycleObject>>

    private var sensorOrientation: Int = 0
    private var previewWidth = 0
    private var previewHeight = 0
    private var computingDetection: Boolean = false

    private var rgbFrameBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null
    private var copiedBitmap: Bitmap? = null

    private var DESIRED_PREVIEW_SIZE = Size(1920, 1080)

    private var MAINTAIN_ASPECT = true
    val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private val dbHelper = DbHelper(this)
    private val db = dbHelper.writableDatabase

    override fun getLayoutId(): Int {
        return R.layout.fragment_camera
    }

    override fun getDesiredPreviewFrameSize(): Size {
        return DESIRED_PREVIEW_SIZE
    }
    @RequiresApi(Build.VERSION_CODES.N)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        liveModel.finalViolationsList.observe(this, { list->
            Log.d("SIZE_LIST", list.size.toString())
            saveDataToDb()
        })
    }
    override fun onPreviewSizeChosen(size: Size, rotation: Int) {

        val textSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, resources.displayMetrics)
        borderedText = BorderedText(textSizePx)
        borderedText.setTypeFace(Typeface.MONOSPACE)

        tracker = BoxTracker(this)

        try{
            detector = DetectorFactory.getDetector(assets, MainActivity.MAIN_MODEL_NAME)
        }catch(e: Exception){
            Toast.makeText(this, "Classifier can't be initiated", Toast.LENGTH_SHORT).show()
            finish()
        }

        val cropSize = detector.getInputSize()

        Log.d("CROP_SIZE", "${detector.getInputSize()}")

        previewWidth = size.width
        previewHeight = size.height
        Log.d("ORIENTATION:", "$rotation AND ${getScreenOrientation()}")
        sensorOrientation = rotation - getScreenOrientation()

        Log.d("CAM_SIZE", "Initializing at $previewWidth x $previewHeight")
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)

        frameToCropMat = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight, cropSize, cropSize, sensorOrientation, MAINTAIN_ASPECT
        )

        cropToFrameMat = Matrix()
        frameToCropMat.invert(cropToFrameMat)

        trackingOverlay = findViewById(R.id.tracking_overlay)
        trackingOverlay.addCallback(
            object: OverlayView.DrawCallback{
                override fun drawCallback(canvas: Canvas) {
                    tracker.draw(canvas)
                    //tracker.drawDebugBoxes(canvas)
                }
            }
        )

        tracker.setFrameConfig(previewWidth, previewHeight, sensorOrientation)
    }

    @RequiresApi(Build.VERSION_CODES.N)
    @Synchronized
    fun saveDataToDb(){
        val violationList = liveModel.getFinalViolationsList()!!
        for(violation in violationList){
            val values = ContentValues().apply{
                put(ViolationEntry.COL_TIMESTAMP, Calendar.getInstance().time.toString())
                put(ViolationEntry.COL_LOC, "Lapu-Lapu City")
                put(ViolationEntry.COL_SNAPSHOT, "thislocation")
                put(ViolationEntry.COL_OFFENSE, violation.offense)
                put(ViolationEntry.COL_LP, if(violation.licensePlate != null) violation.licensePlate!!.licenseNumber else "")
            }
            val newRowId = db.insert(ViolationEntry.TABLE_NAME, null, values)
            Log.d("DATABASE_OP", "Inserted $newRowId")
            liveModel.removeFromFinalViolationsList(violation)
        }
    }

    @RequiresApi(Build.VERSION_CODES.N)
    override fun processImage() {
        trackingOverlay.postInvalidate()

        if(computingDetection){
            readyForNextImage()
            return
        }
        computingDetection = true
        Log.d("Processing Image", "$previewWidth x $previewHeight")
        rgbFrameBitmap!!.setPixels(getRGBBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight)

        readyForNextImage()

        val canvas = Canvas(croppedBitmap!!)
        canvas.drawBitmap(rgbFrameBitmap!!, frameToCropMat, null)

        //ImageUtils.saveBitmap(rgbFrameBitmap!!, "preview.png")

        runInBackground {
            val results: List<Detection> = detector.detectImage(croppedBitmap!!)
            copiedBitmap = Bitmap.createBitmap(croppedBitmap!!)
            val canvas1 = Canvas(copiedBitmap!!)
            val paint = Paint()
            paint.color = Color.BLUE
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.0f

            val minConfidence = MainActivity.MINIMUM_CONFIDENCE

            var mappedPredictions: MutableList<Detection> = LinkedList()

            for (result in results) {
                val location: RectF = result.location

                Log.d("DETECTION",result.detectedClass.toString())

                if (location != null && result.confidence >= minConfidence) {
                    canvas1.drawRect(location, paint)
                    cropToFrameMat.mapRect(location)
                    result.location = location
                    mappedPredictions.add(result)
                }
            }
            tracker.trackResults(mappedPredictions, 0)
            trackingOverlay.postInvalidate()

            // Detect Violations
            detectViolations(mappedPredictions)

            computingDetection = false
        }
    }

    @RequiresApi(Build.VERSION_CODES.N)
    private fun detectViolations(results: MutableList<Detection>) {
        /** Logic will be:
         * 1.) for result in results
         * 2.) if object is class motorcylist(0), create new object predictedMotorcyclist
         * 3.) since results is ordered by 0, 0, 1, 1, 2, 2, etc... then we can assume that the number of motorcyclist
         *      is what we are concerned with.
         * 4.) if object is class 1, create new helmetList and add this object with the same id
         * 5.) if object is class 2, create new license plate and add this object
         * 6.) Check each of their rects: If helmet rect is within motorcycle rect, add it to that predictedMotorycle
         *      object. Same with license plate
         * 7.) If rect is not, then discard
         * 8.) Finally, check each motorcycle object if their helmet has been initialized. Basically, if a motorcycle
         *      has a null helmet, then that motorcycle might not have a helmet. Add this to list of potential flags.
         *      Maybe add a license plate checker on the next frame to reconfirm if this object really does not have
         *      a helmet.
         */
        try{
            var motorcycleList: MutableList<MotorcycleObject> = ArrayList()
            var helmetList: MutableList<Detection> = ArrayList()
            var lpList: MutableList<Detection> = ArrayList()
            var tricycleList: MutableList<Detection> = ArrayList()

            var index = 0
            var helmetIndex = 0
            var lpIndex = 0
            for(result in results){
                if(result.detectedClass == 0){
                    val motorcycleObject = MotorcycleObject()
                    result.id = index++.toString()
                    motorcycleObject.motorcyclist = result
                    motorcycleList.add(motorcycleObject)
                    Log.d("MOTOR_DETECTED:", motorcycleList.size.toString())
                }

                if(result.detectedClass == 1){
                    //result.id = helmetIndex++.toString()
                    helmetList.add(result)
                }
                if(result.detectedClass == 2){
                    //result.id = lpIndex++.toString()
                    lpList.add(result)
                }
                if(result.detectedClass == 3){
                    tricycleList.add(result)
                }
            }
            /**
             * Check each motorcycle for the location rect, and see if any helmet or lp is within this rectangle.
             */
            for(motorcycle in motorcycleList){
                var isTricycle = false
                // Check if motorcycle is bound within a tricycle box. The requirements are the motorcycle must be
                // Within the rects of half of the left side of the tricycle. Otherwise, it's most likely a tricycle.
                for(tricycle in tricycleList){
                    if(motorcycle.motorcyclist!!.location.intersects(
                            tricycle.location.left,
                            tricycle.location.top,
                            tricycle.location.right,
                            tricycle.location.bottom
                        )){
                        isTricycle = true
                        break
                    }
                }
                if(isTricycle){
                    Log.d("TRICYCLE", "Detection is a Tricycle")
                    //Skip everything below if isTricycle
                    continue
                }else{
                    Log.d("TRICYCLE", "Detection is not a tricycle")
                }
                for(helmet in helmetList){
                    if(helmet.location.intersects(
                            motorcycle.motorcyclist!!.location.left,
                            motorcycle.motorcyclist!!.location.top,
                            motorcycle.motorcyclist!!.location.right,
                            motorcycle.motorcyclist!!.location.bottom)){

                        motorcycle.helmet = helmet
                        break
                    }
                }
                for(lp in lpList){
                    if(lp.location.intersects(
                            motorcycle.motorcyclist!!.location.left,
                            motorcycle.motorcyclist!!.location.top,
                            motorcycle.motorcyclist!!.location.right,
                            motorcycle.motorcyclist!!.location.bottom)) {

                        motorcycle.licensePlate = LicensePlate(lp)
                        motorcycle.snapshot = rgbFrameBitmap!!
                        readLp(motorcycle)
                        break
                    }
                }
                /**
                 * Check current potential violations list. If there is any, process the violations list.
                 */
                val originalPotentialViolationsList = liveModel.getPotentialViolationsList()
                for(potViolator in originalPotentialViolationsList!!){
                    var potAddedToFinal = false
                    for(lp in lpList){
                        /**
                         * TEMPORARY
                         */
                        // IF THIS IS TRUE, THEN CONFIRM THE VIOLATION
                        if(potViolator.potentialViolation && lp.location.intersects(
                                potViolator.motorcyclist!!.location.left,
                                potViolator.motorcyclist!!.location.top,
                                potViolator.motorcyclist!!.location.right,
                                potViolator.motorcyclist!!.location.bottom)) {
                            // Remove from potential violations list
                            liveModel.removeFromPotentialViolationsList(potViolator)
                            // Call and prepare to save data in device or send over the network
                            processViolation(potViolator)
                            Log.d("MOTOR_DETECTED", "NO HELMET FINAL VERDICT")
                            potAddedToFinal = true
                            break
                        }
                    }
                    // If none of the current lps' intersect with a previous potential violator, remove it from
                    // The list.
                    if(!potAddedToFinal){
                        liveModel.removeFromPotentialViolationsList(potViolator)
                        Log.d("NO_VIOL", "No violators detected within current frame")
                    }
                }

                /**
                 * Place motorcycle into potential violation. Next Frame, check if existing license plate is within potential
                 * violation list.
                 */
                if(motorcycle.helmet == null && !motorcycle.potentialViolation && !motorcycle.finalViolation){
                    Log.d("MOTOR_DETECTED", "HELMET NOT FOUND, PLACING IN POTENTIAL")
                    motorcycle.potentialViolation = true
                    motorcycle.offense = "no helmet"
                    liveModel.addToPotentialViolationsList(motorcycle)
                }
            }
        }catch(e: Exception){
            e.printStackTrace()
        }
    }

    /**
     * Process the final violations list. Save it into device, preparing for upload
     */

    private fun processViolation(potViolator: DetectorActivity.MotorcycleObject) {
        if(potViolator.licensePlate != null){
            Runnable{
                readLp(potViolator)
                if(!liveModel.getReportedList()!!.contains(potViolator.licensePlate!!.licenseNumber)){
                    Log.d("PROC_VIOL", "ADDING LICENSE" )
                    liveModel.addToReportedList(potViolator.licensePlate!!.licenseNumber)
                    liveModel.addToFinalViolationsList(potViolator)
                }else{
                    Log.d("PROC_VIOL", "LICENSE ALREADY PREPARED FOR REPORT" )
                }
            }.run()
            /**
             * 1.) SAVE DATA TO DEVICE. PREPARE TO UPLOAD OVER THE NETWORK
             * 1.) ADD CHECKER FOR NETWORK CONFIG. IF NETWORK WORKS
             */
        }else{
            Log.d("PROC_VIOL", "No LP detected")
        }

    }
    /**
     * Checker for LP
     */
    @Synchronized
    fun readLp(motorcycleObject: MotorcycleObject){
        val licensePlate = motorcycleObject.licensePlate!!.detectedLicense
        /**
         * Get the original bitmap rgbFrameBitmap, then use RectF to crop bitmap. Save bitmap after.
         */
        var croppedLicensePlate = Bitmap.createBitmap(licensePlate.location.width().toInt(),
            licensePlate.location.height().toInt(), Bitmap.Config.ARGB_8888)

        //Create new canvas
        val canvas = Canvas(croppedLicensePlate)

        //Draw the background
        val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        paint.color = Color.WHITE
        canvas.drawRect(
            Rect(0,
                0,
                licensePlate.location.width().toInt(),
                licensePlate.location.height().toInt()),
            paint)

        val matrix = Matrix()
        matrix.postTranslate(-licensePlate.location.left, -licensePlate.location.top)
        //canvas.save()
        //canvas.rotate(90f)
        canvas.drawBitmap(motorcycleObject.snapshot!!, matrix, paint)
        //canvas.restore()
       // matrix.postRotate(90f)
       // croppedLicensePlate = Bitmap.createBitmap(croppedLicensePlate, 0, 0, croppedLicensePlate.width, croppedLicensePlate.height, matrix, true )
        //ImageUtils.saveBitmap(croppedLicensePlate, "lp.png")

        // USE ML KIT TEXT RECOGNIZER
        val result = recognizer.process(InputImage.fromBitmap(croppedLicensePlate, 90))
            .addOnSuccessListener {
                val resultTexts = it.textBlocks
                for(text in resultTexts){
                    val replace = text.text.replace("\\s".toRegex(), "")
                    // Valid Philippine License Plate Numbers length, both Temporary and Permanent
                    if(replace.length == 6 || replace.length == 7 || replace.length == 11 || replace.length == 12 ){
                        motorcycleObject.licensePlate!!.licenseNumber = text.text
                    }
                }
            }
            .addOnFailureListener {e->
                Log.d("LP_READ", e.toString())
            }

    }

    private fun getScreenOrientation(): Int{
        // If android version is equal or above R
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.R){
            (return when(this.display!!.rotation){
                Surface.ROTATION_270 -> 270
                Surface.ROTATION_180 -> 180
                Surface.ROTATION_90 -> 90
                else -> 0
            })
        }else{
            @Suppress("DEPRECATION")
            (return when (windowManager.defaultDisplay.rotation) {
                Surface.ROTATION_270 -> 270
                Surface.ROTATION_180 -> 180
                Surface.ROTATION_90 -> 90
                else -> 0
            })
        }
    }

    companion object{
        const val TEXT_SIZE_DIP = 10.0f
    }

    class MotorcycleObject {
        var snapshot: Bitmap? = null
        var motorcyclist: Detection? = null
        var helmet: Detection? = null
        var licensePlate: LicensePlate? = null
        var potentialViolation: Boolean = false
        var finalViolation: Boolean = false
        var offense: String = ""

        // Pass license plate recognition to a separate reader. Maintain id as identifier for which motorcycle owns which
        // license plate
    }

    class LicensePlate {

        var licenseNumber: String = ""
        var detectedLicense = Detection()

        constructor(detection: Detection){
            detectedLicense = detection
        }

    }
}
