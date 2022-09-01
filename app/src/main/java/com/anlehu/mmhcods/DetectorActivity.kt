/************************************************************************************************************
 * DetectorActivity - Activity that handles the image processing and detection features
 ************************************************************************************************************/

package com.anlehu.mmhcods

import android.content.ContentValues
import android.database.sqlite.SQLiteDatabase
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
import com.anlehu.mmhcods.utils.*
import com.anlehu.mmhcods.utils.DbHelper.ViolationEntry
import com.anlehu.mmhcods.utils.Detector.Detection
import com.anlehu.mmhcods.views.OverlayView
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.util.*

open class DetectorActivity: CameraActivity(), ImageReader.OnImageAvailableListener {

    private lateinit var trackingOverlay: OverlayView
    private lateinit var frameToCropMat: Matrix
    private lateinit var cropToFrameMat: Matrix
    private lateinit var frameToLaneMat: Matrix
    private lateinit var laneToFrameMat: Matrix
    private lateinit var borderedText: BorderedText
    private lateinit var tracker: BoxTracker
    private lateinit var detector: YoloV5Classifier
    private lateinit var laneDetector: LaneClassifier

    private lateinit var finViolationObserver: androidx.lifecycle.Observer<MutableList<MotorcycleObject>>

    private var sensorOrientation: Int = 0
    private var previewWidth = 0
    private var previewHeight = 0
    private var computingDetection: Boolean = false

    private var croppedBitmap: Bitmap? = null
    private var laneBitmap: Bitmap? = null
    private var copiedBitmap: Bitmap? = null

    private var DESIRED_PREVIEW_SIZE = Size(1080, 1920)

    private var MAINTAIN_ASPECT = true

    private val dbHelper = DbHelper(this)
    private lateinit var db: SQLiteDatabase

    var mountedPath = ""

    /********************************************************************************************************
     * Gets layout id of the camera fragment
     * @return Int ID value
     ********************************************************************************************************/
    override fun getLayoutId(): Int {
        return R.layout.fragment_camera
    }
    /********************************************************************************************************
     * Gets the desired preview frame size
     * @return Size value of desired preview size
     ********************************************************************************************************/
    override fun getDesiredPreviewFrameSize(): Size {
        return DESIRED_PREVIEW_SIZE
    }
    /********************************************************************************************************
     * On create method
     ********************************************************************************************************/
    @RequiresApi(Build.VERSION_CODES.N)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mountedPath = intent.extras!!.getString("path").toString()
        createDatabase()
    }


    @RequiresApi(Build.VERSION_CODES.N)
    private fun createDatabase(){
        db = dbHelper.writableDatabase
        liveModel.finalViolationsList.observe(this) { list ->
            //Log.d("SIZE_LIST", list.size.toString())
            saveDataToDb()
        }
    }

    private fun closeDb(){
        dbHelper.close()
    }

    /********************************************************************************************************
     * Create a table if it does not exist
     ********************************************************************************************************/

    /********************************************************************************************************
     * Function that saves data to database
     ********************************************************************************************************/
    @RequiresApi(Build.VERSION_CODES.N)
    @Synchronized
    fun saveDataToDb(){
        val violationList = liveModel.getFinalViolationsList()!!
        for(violation in violationList){
            val values = ContentValues().apply{
                put(ViolationEntry.COL_TIMESTAMP, Calendar.getInstance().time.toString())
                put(ViolationEntry.COL_LOC, "Lapu-Lapu City")
                put(ViolationEntry.COL_SNAPSHOT, "thislocation")
                put(ViolationEntry.COL_HELMET, violation.offenseHelmet)
                put(ViolationEntry.COL_COUNTERFLOW, violation.offenseCounterflow)
                put(ViolationEntry.COL_LP, if(violation.licensePlate != null) violation.licensePlate!!.licenseNumber else "")
            }
            val newRowId = db.insert(ViolationEntry.TABLE_NAME, null, values)
            //Log.d("DATABASE_OP", "Inserted $newRowId")
            liveModel.removeFromFinalViolationsList(violation)
        }
    }
    /********************************************************************************************************
     * Override onPreviewSizeChosen method; Creates detector objects used for object detection;
     * Sets image templates and matrices with correct sizes to fit the detector requirements;
     * Adds a callback to a drawing overlay canvas that draws boxes if any detection occurs
     ********************************************************************************************************/
    override fun onPreviewSizeChosen(size: Size, rotation: Int) {
        try{
            // mountedPath is null for some reason.
            //Log.d("MOUNTED_PATH", mountedPath)
            detector = DetectorFactory.getDetector(mountedPath, ModelUtils.MAIN_MODEL_NAME)
            laneDetector = DetectorFactory.getLaneDetector(mountedPath, ModelUtils.LANE_MODEL_NAME)
        }catch(e: Exception){
            //Log.d("Classifier", "$e")
            Toast.makeText(this, "Classifier/s can't be initiated", Toast.LENGTH_SHORT).show()
            finish()
        }

        val textSizePx = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, resources.displayMetrics)
        borderedText = BorderedText(textSizePx)
        borderedText.setTypeFace(Typeface.MONOSPACE)

        tracker = BoxTracker(this)

        val cropSize = detector.getInputSize()
        val inputWidth = laneDetector.INPUT_WIDTH
        val inputHeight = laneDetector.INPUT_HEIGHT

        //Log.d("CROP_SIZE", "${detector.getInputSize()}")

        previewWidth = 1920
        previewHeight = 1080
//        previewWidth = 1280
//        previewHeight = 720
        //Log.d("ORIENTATION:", "$rotation AND ${getScreenOrientation()}")
        sensorOrientation = rotation - getScreenOrientation()

        //Log.d("CAM_SIZE", "Initializing at $previewWidth x $previewHeight")
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        laneBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)

        frameToCropMat = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight, cropSize, cropSize, sensorOrientation, MAINTAIN_ASPECT
        )

        frameToLaneMat = ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight, inputWidth, inputHeight, sensorOrientation, MAINTAIN_ASPECT
        )

        cropToFrameMat = Matrix()
        frameToCropMat.invert(cropToFrameMat)

        laneToFrameMat = Matrix()
        frameToLaneMat.invert(laneToFrameMat)

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
    /********************************************************************************************************
     * Processes image for detection
     ********************************************************************************************************/

    @RequiresApi(Build.VERSION_CODES.O)
    override fun processImage() {

        trackingOverlay.postInvalidate()    // invalidate tracking overlay

        if(computingDetection){             // if computingDetection is true
            readyForNextImage()             // ready next image
            return                          // return from function
        }
        computingDetection = true           // set computingDetection to true
        //Log.d("Processing Image", "$previewWidth x $previewHeight")
        // set pixels of rgbFrameBitmap
        rgbFrameBitmap!!.setPixels(getRGBBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight)

        readyForNextImage()                 // close current image and prepare for next frame

        //val canvas = Canvas(croppedBitmap!!)    // create canvas object using croppedBitmap
        //croppedBitmap = ImageUtils.rescaleImage(rgbFrameBitmap, Size(640, 640),"", 0f, true)
        //croppedBitmap = Bitmap.createScaledBitmap(rgbFrameBitmap!!, 640, 640, true)
        croppedBitmap = ImageUtils.rescaleImage(rgbFrameBitmap, Size(640, 640), "", 0f, true)
        laneBitmap = ImageUtils.rescaleImage(rgbFrameBitmap, Size(800, 288), "", 90f, true)


        // draw on canvas through matrix using rgbFrameBitmap as data values, which will be drawn on croppedBitmap
        //canvas.drawBitmap(rgbFrameBitmap!!, frameToCropMat, null)
        ImageUtils.saveBitmap(croppedBitmap!!, "supposed")

        //ImageUtils.saveBitmap(rgbFrameBitmap!!, "preview.png")

        // run in a background thread
        runInBackground {
            val results: List<Detection> = detector.detectImage(croppedBitmap!!)
//            for(i in results){
//                val str = "left: ${i.location.left} | top: ${i.location.top} | right: ${i.location.right} | bottom: ${i.location.bottom}"
//                when (i.detectedClass) {
//                    0 -> Log.d("MOTOR LOC", str)
//                    1 -> Log.d("HELMET LOC", str)
//                    2 -> Log.d("LP LOC", str)
//                    else -> Log.d("TRI LOC", str)
//                }
//            }
            //TODO: Make Lane Detector like how the YoloV5 works
            val laneResults  = laneDetector.detectImage(laneBitmap!!)
            copiedBitmap = Bitmap.createBitmap(croppedBitmap!!)
            val canvas1 = Canvas(copiedBitmap!!)
            val paint = Paint()
            paint.color = Color.BLUE
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.0f

            val minConfidence = ModelUtils.MINIMUM_CONFIDENCE // sets minimum confidence levels for detection

            var mappedPredictions: MutableList<Detection> = mutableListOf()    // create a list for detection map

            /*
                Go through every result of the object detection
             */
            for (result in results) {
                val location: RectF = result.location           // get RectF location of detected object in image if any

                //Log.d("DETECTION",result.detectedClass.toString())

                // if location is not null and the detection confidence is over threshold, then...
                if (result.confidence >= minConfidence) {
                    //canvas1.drawRect(location, paint)           // draw on canvas1 for the location of detected object
                   // cropToFrameMat.mapRect(location)            // add location rects to cropToFrameMat
                    //result.location = location
                    mappedPredictions.add(result)               // add result to mappedPredictions
                }
            }
            /** Temp**/
            tracker.trackResults(mappedPredictions, laneResults, 0)    // add mappedPredictions to tracker overlay
            trackingOverlay.postInvalidate()

            // Detect Violations
            detectViolations(mappedPredictions, laneResults)

            computingDetection = false
        }
    }
    /********************************************************************************************************
     * Retrieves screen orientation
     ********************************************************************************************************/
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
    /********************************************************************************************************
     * singleton tied to detector activity
     ********************************************************************************************************/
    companion object{

        val outputRangeX = floatArrayOf(0f, 1920f)
        val outputRangeY = floatArrayOf(0f, 1080f)

        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

        var rgbFrameBitmap: Bitmap? = null
        const val TEXT_SIZE_DIP = 10.0f

        val liveModel = DataViewModel()

        /********************************************************************************************************
         * Detects violations within a frame
         * @param results - the result list of detections within a frame
         ********************************************************************************************************/

        @RequiresApi(Build.VERSION_CODES.O)
        fun detectViolations(results: MutableList<Detection>, laneResults: MutableList<FloatArray>) {

            /** Logic will be:
             * 1.) for result in results
             * 2.) if object is class motorcylist(0), create new object predictedMotorcyclist
             * 3.) since results is ordered by 0, 0, 1, 1, 2, 2, etc... then we can assume that the number of motorcyclist
             *      is what we are concerned with.
             * 4.) if object is class 1, create new helmetList and add this object with the same id
             * 5.) if object is class 2, create new license plate and add this object
             * 6.) Check each of their rects: If helmet rect is within upper 1/3 of motorcycle rect, add it to that
             * predictedMotorycle object. Same with license plate. Don't consider tricycles.
             * 7.) If rect is not, then discard
             * 8.) Finally, check each motorcycle object if their helmet has been initialized. Basically, if a motorcycle
             *      has a null helmet, then that motorcycle might not have a helmet. Add this to list of potential flags.
             *      Add a license plate checker on the next frame to reconfirm if this object really does not have
             *      a helmet.
             */
            try{
                var motorcycleList: MutableList<MotorcycleObject> = ArrayList()
                var helmetList: MutableList<Detection> = ArrayList()
                var lpList: MutableList<Detection> = ArrayList()
                var tricycleList: MutableList<Detection> = ArrayList()

                var index = 0
                //var helmetIndex = 0
                //var lpIndex = 0
                for(result in results){
                    if(result.detectedClass == 0){                                  // detect if motorcycle
                        val motorcycleObject = MotorcycleObject()
                        result.id = index++.toString()                              // increment index identifier
                        motorcycleObject.motorcyclist = result                      // set the motorcyclist object
                        motorcycleObject.dateTime = FileUtil.DateToString(FileUtil.getDateTime())          // set datetime of violation
                        motorcycleList.add(motorcycleObject)                        // add moto object to list
                        //Log.d("MOTOR_DETECTED:", motorcycleList.size.toString())
                    }

                    if(result.detectedClass == 1){
                        //result.id = helmetIndex++.toString()
                        helmetList.add(result)                                      // add current helmet result in list
                    }
                    if(result.detectedClass == 2){
                        //result.id = lpIndex++.toString()
                        lpList.add(result)                                          // add current lp in list
                    }
                    if(result.detectedClass == 3){
                        tricycleList.add(result)                                    // add current tricycle in list
                        //Log.d("TRIKE_LOC", "Found Tricycle")
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

                        //Log.d("TRICYCLE_LOC", "${tricycle.location.right}, ${tricycle.location.top}")
                        // Create new Rect for tricycle consideration threshold
                        val thresholdPoint = (tricycle.location.top - tricycle.location.bottom)/2

                        if(motorcycle.motorcyclist!!.location.intersects(
                                tricycle.location.left,
                                thresholdPoint,
                                tricycle.location.right,
                                tricycle.location.bottom
                            )){
                            isTricycle = true
                            break
                        }
                    }
                    if(isTricycle){
                        //Log.d("TRICYCLE", "Detection is a Tricycle")
                        //Skip everything below if isTricycle
                        continue
                    }
                    for(helmet in helmetList){
                        // Check if helmet is within upper 1/3 of motorcycle rect. Image is in landscape
                        if(helmet.location.intersects(
                                motorcycle.motorcyclist!!.location.left/3,
                                motorcycle.motorcyclist!!.location.top,
                                motorcycle.motorcyclist!!.location.right/3,
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

                    /** Check rect locations for overtaking/counterflowing
                     */

                    if(laneResults.size != 0){
                        var pointX = 0
                        if(motorcycle.licensePlate != null)
                            pointX = (motorcycle.licensePlate!!.detectedLicense.location.left.toInt() + (motorcycle.licensePlate!!.detectedLicense.location.width().toInt() / 2))
                         else{
                            pointX = (motorcycle.motorcyclist!!.location.left.toInt() + (motorcycle.motorcyclist!!.location.width().toInt() / 2))
                        }
                        val pointY = -motorcycle.motorcyclist!!.location.bottom.toInt()
                        val pointOfReference = Point(pointX, pointY)

                        // Check if this point is within the path

                        var index = if(laneResults.size > 1){
                            1
                        }else{
                            0
                        }

                        val a = Point(laneResults[index][2].toInt(), -laneResults[index][3].toInt())
                        val b = Point(laneResults[index][0].toInt(), -laneResults[index][1].toInt())
//                        Log.d("COUNTERFLOW_LANE", "(${b.x}, ${b.y}) , (${a.x}, ${a.y})")
                        val determinant = (b.y - a.y)*(pointOfReference.x - a.x) - (b.x - a.x)*(pointOfReference.y - a.y)
                        Log.d("Determinant", "$determinant")
                        if(determinant > 0){
                            // counterflow detected
                            motorcycle.offenseCounterflow = "yes"
                        }
                    }

                    /**
                       * Check current potential violations list. If there is any, process the violations list.
                     */
                    val originalPotentialViolationsList = liveModel.getPotentialViolationsList() // Get potential list
                    var refTime = FileUtil.getDateTime()
                    for(potViolator in originalPotentialViolationsList!!){                      // check every pot vio
                        var potAddedToFinal = false
                        for(motorcycle in motorcycleList){
                            // check if lp of potential violator exists within detected motorcycle objects
                            // IF THIS IS TRUE, THEN CONFIRM THE VIOLATION.
                            if(potViolator.potentialViolation){
                               if((potViolator.motorcyclist!!.location.contains(motorcycle.licensePlate!!.detectedLicense.location)
                                            || potViolator.licensePlate!!.licenseNumber.equals(motorcycle.licensePlate!!.licenseNumber, true)
                                            || ModelUtils.box_iou(potViolator.motorcyclist!!.location, motorcycle.motorcyclist!!.location) < 0.9)) {
                                   // If previous violator is still in this frame,
                                   // after 2 seconds, process it to final violation. Checking involves
                                   // looking at the relative locations of the motorcyclist, LP or a License Number match
                                   // and checking if they have the same data
                                   motorcycle.potentialViolation = true        // motorcycle has been found in pot vio list
                                       if ((FileUtil.elapsedTime(FileUtil.stringToDate(potViolator.dateTime), refTime) >= 2.0)) {
                                           // Remove from potential violations list
                                           liveModel.removeFromPotentialViolationsList(potViolator)
                                           // Call and prepare to save data in device
                                           motorcycle.potentialViolation = false
                                           motorcycle.finalViolation = true
                                           processViolation(potViolator)
                                           potAddedToFinal = true
                                           break
                                   }
                               }
                            }
                        }
                        // If none of the current lps' intersect with a previous potential violator, remove it from
                        // The list if enough time has elapsed.
                        if(!potAddedToFinal){
                            if((FileUtil.elapsedTime(FileUtil.stringToDate(potViolator.dateTime), refTime) >= 2.0)){
                                liveModel.removeFromPotentialViolationsList(potViolator)
                                //Log.d("NO_VIOL", "No violators detected within current frame")
                            }
                        }
                    }
                    /**
                     * Place motorcycle into potential violation if non-existent yet. Next Frame, check if existing license plate is within potential
                     * violation list. (Code above)
                     * NO HELMET DETECTION
                     */

                    if(!motorcycle.potentialViolation && !motorcycle.finalViolation){

                        motorcycle.potentialViolation = true
                        if(motorcycle.helmet == null){
                            //Log.d("MOTOR_DETECTED", "HELMET NOT FOUND, PLACING IN POTENTIAL")
                            motorcycle.offenseHelmet = "yes"
                        }
                        // Add to potential Violations List
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
                    //readLp(potViolator) redundant?
                    if(!liveModel.getReportedList()!!.contains(potViolator.licensePlate!!.licenseNumber)){
                        //Log.d("PROC_VIOL", "ADDING LICENSE ${potViolator.licensePlate!!.licenseNumber}" )
                        liveModel.addToReportedList(potViolator.licensePlate!!.licenseNumber)
                        potViolator.finalViolation = true
                        liveModel.addToFinalViolationsList(potViolator)
                    }else{
                        //Log.d("PROC_VIOL", "LICENSE ALREADY PREPARED FOR REPORT" )
                    }
                }.run()
                /**
                 * 1.) SAVE DATA TO DEVICE. PREPARE TO UPLOAD OVER THE NETWORK
                 * 1.) ADD CHECKER FOR NETWORK CONFIG. IF NETWORK WORKS
                 */
            }else{
                //Log.d("PROC_VIOL", "No LP detected")
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
                            //Log.d("LP_READ", text.text)
                            motorcycleObject.licensePlate!!.licenseNumber = text.text
                        }
                    }
                }
                .addOnFailureListener {e->
                    //Log.d("LP_READ", e.toString())
                }

        }
    }

    class MotorcycleObject {
        var snapshot: Bitmap? = null
        var motorcyclist: Detection? = null
        var helmet: Detection? = null
        var licensePlate: LicensePlate? = null
        var potentialViolation: Boolean = false
        var finalViolation: Boolean = false
        var offenseHelmet: String = "no"
        var offenseCounterflow: String = "no"
        var dateTime: String = ""
    }

    class LicensePlate {

        var licenseNumber: String = ""
        var detectedLicense = Detection()

        constructor(detection: Detection){
            detectedLicense = detection
        }

    }
}
