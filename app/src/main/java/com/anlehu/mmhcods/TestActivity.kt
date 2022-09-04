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
import android.util.Pair
import android.util.Size
import android.widget.Button
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.anlehu.mmhcods.utils.*
import java.io.File
import java.time.format.DateTimeFormatter
import java.util.*

/**
 * Performs violation testing for research/graphing purposes
 */

class TestActivity : AppCompatActivity() {

    private var handlerThread: HandlerThread? = null
    private var handler: Handler? = null

    private lateinit var dirButton: Button
    private lateinit var frameToCropMat: Matrix
    private lateinit var cropToFrameMat: Matrix
    private lateinit var frameToLaneMat: Matrix
    private lateinit var laneToFrameMat: Matrix
    private lateinit var borderedText: BorderedText
    private lateinit var tracker: BoxTracker
    private lateinit var detector: YoloV5Classifier
    private lateinit var laneDetector: LaneClassifier

    private var previewWidth = 0
    private var previewHeight = 0
    private var computingDetection: Boolean = false

    private var rgbFrameBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null
    private var laneBitmap: Bitmap? = null
    private var copiedBitmap: Bitmap? = null

    private var MAINTAIN_ASPECT = true

    private var mountedPath = ""

    // Box Tracking
    private var trackedObjects: MutableList<TrackedPrediction> = LinkedList()
    private var boxPaint: Paint = Paint()
    private var frameWidth: Int = 0
    private var frameHeight: Int = 0
    private var sensorOrientation: Int = 0

    private var screenRects: MutableList<Pair<Float, RectF>> = LinkedList()
    private lateinit var lanePoints: FloatArray

    var frameToCanvasMat: Matrix = Matrix()

    private class TrackedPrediction{
        lateinit var location: RectF
        lateinit var lanePoints: FloatArray
        var detectionConfidence: Float = 0.0f
        var color: Int = 0
        var title = ""
        var id: Int = 0
    }


    /********************************************************************************************************
     * Starts activity for result
     ********************************************************************************************************/
    @RequiresApi(Build.VERSION_CODES.O)
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
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?){
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_test)
        mountedPath = intent.extras!!.getString("path").toString()

        boxPaint.color = Color.RED
        boxPaint.strokeWidth = 10.0f
        boxPaint.strokeCap = Paint.Cap.ROUND
        boxPaint.strokeJoin = Paint.Join.ROUND
        boxPaint.strokeMiter = 100f
        boxPaint.style = Paint.Style.STROKE

        setFrameConfig(1920, 1080, 90)

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
    @RequiresApi(Build.VERSION_CODES.O)
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
        var files: Array<File> = directory.listFiles() as Array<File>
        Log.d("FILES_SIZE", "Size: ${files.size}")

        // Check if location contains jpeg


        for(item in files){
            if(item.toString().endsWith(".jpg") or item.toString().endsWith(".png")
                or item.toString().endsWith(".jpeg")){
                Log.d("FILE_ITEM", "$item")
                val options = BitmapFactory.Options()
                options.inPreferredConfig = Bitmap.Config.ARGB_8888
                val loadedBitmap = BitmapFactory.decodeFile(item.path, options)
                rgbFrameBitmap = ImageUtils.rescaleImage(loadedBitmap,Size(1920, 1080), item.path, 0f, true)
                croppedBitmap = ImageUtils.rescaleImage(loadedBitmap,Size(640, 640), item.path, 90f, true)
                laneBitmap = ImageUtils.rescaleImage(loadedBitmap,Size(800, 288), "lane${item.path}", 0f, true)

                ImageUtils.saveBitmap(croppedBitmap!!, "supposed")

                ImageUtils.saveBitmap(rgbFrameBitmap!!, "preview")

                /*
                    Apply testing per image
                */
                processImage()
            }
        }


    }

    @Synchronized
    fun runInBackground(r: Runnable){
        if(handler != null)
            handler!!.post(r)
    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun processImage(){
        // run in a background thread
        //runInBackground {
            val bitmap = rgbFrameBitmap
            val results: List<Detector.Detection> = detector.detectImage(croppedBitmap!!)

            for(i in results){
                val str = "left: ${i.location.left} | top: ${i.location.top} | right: ${i.location.right} | bottom: ${i.location.bottom}"
                when (i.detectedClass) {
                    0 -> Log.d("MOTOR LOC", str)
                    1 -> Log.d("HELMET LOC", str)
                    2 -> Log.d("LP LOC", str)
                    else -> Log.d("TRI LOC", str)
                }
            }

            val laneResults  = laneDetector.detectImage(laneBitmap!!)
            copiedBitmap = Bitmap.createBitmap(croppedBitmap!!)
//            val paint = Paint()
//            paint.color = Color.BLUE
//            paint.style = Paint.Style.STROKE
//            paint.strokeWidth = 2.0f

            val minConfidence = ModelUtils.MINIMUM_CONFIDENCE // sets minimum confidence levels
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
            detectViolations(mappedPredictions, laneResults)
            trackResults(mappedPredictions, laneResults, 0)
            var canvas = Canvas(bitmap!!)
            draw(canvas)
            ImageUtils.saveBitmap(bitmap!!, FileUtil.dateToString(DateTimeFormatter.ofPattern("yyyyMMddHHmmss")))

            computingDetection = false
        //}
    }

    @Synchronized
    fun setFrameConfig(
        width: Int, height: Int, sensorOrientation: Int
    ) {
        frameWidth = width
        frameHeight = height
        this.sensorOrientation = sensorOrientation
//        lanePoints = FloatArray(0)
//
//        val rotated = sensorOrientation % 180 == 90
//        val multiplier = Math.min(
//            width / (if (rotated) frameWidth.toFloat() else frameHeight.toFloat()),
//            height / (if (rotated) frameHeight.toFloat() else frameWidth.toFloat())
//        )
//        frameToCanvasMat = ImageUtils.getTransformationMatrix(
//            frameWidth,
//            frameHeight,
//            (multiplier * if (rotated) frameHeight else frameWidth).toInt(),
//            (multiplier * if (rotated) frameWidth else frameHeight).toInt(),
//            sensorOrientation,
//            true
//        )
        frameToCanvasMat = Matrix()
    }

    @Synchronized
    fun draw(canvas: Canvas) {

        for (recognition in trackedObjects) {
            if(recognition.id == 1){
                boxPaint.color = Color.YELLOW
                //Log.d("LANE_DET", "Lane = ${recognition.lanePoints.joinToString(",")}")
                canvas.drawLines(recognition.lanePoints, boxPaint)
            }else{
                val trackedPos = RectF(recognition.location)
                //val trackedPos = RectF(0f, 0f, 1920f, 1080f)
                boxPaint.color = recognition.color
                val cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f
                canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint)
//                val labelString = if (!TextUtils.isEmpty(recognition.title)) String.format(
//                    "%s %.2f", recognition.title,
//                    100 * recognition.detectionConfidence
//                ) else String.format("%.2f", 100 * recognition.detectionConfidence)
//                //            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
//                // labelString);
            }
        }
        //canvas.drawLines(lanePoints, boxPaint)

    }

    @Synchronized
    fun trackResults(results: List<Detector.Detection>, lanes: List<FloatArray>, timeStamp: Long){
        // rectangles to track
        val rectsToTrack: MutableList<Pair<Float, Detector.Detection>> = LinkedList()

        // clear current drawings on screen
        screenRects.clear()

        // grab values based on frame to canvas matrix conversion
        //var rgbFrameToScreen = Matrix(frameToCanvasMat)

        for(result in results){

            val detFrameRect = RectF(result.location)
//            Log.d("NMS_TRACK", "Left: ${result.location.left} | Right: ${result.location.right} | Top: ${result.location.top} | " +
//                    "Bottom: ${result.location.bottom}")
            val detScreenRect = RectF()
            //rgbFrameToScreen.mapRect(detScreenRect, detFrameRect)
            screenRects.add(Pair<Float, RectF>(result.confidence, detScreenRect))
            rectsToTrack.add(Pair<Float, Detector.Detection>(result.confidence, result))

            if(detFrameRect.width() < BoxTracker.MIN_SIZE ||detFrameRect.height() < BoxTracker.MIN_SIZE){
                continue
            }

            rectsToTrack.add(Pair<Float, Detector.Detection>(result.confidence, result))
        }

        trackedObjects.clear()
        if(rectsToTrack.isEmpty() && lanes.isEmpty()){
            // Abort because there's nothing to track
            return
        }

        val colors: Array<Int> = arrayOf(
            Color.BLUE,
            Color.RED,
            Color.GREEN,
            Color.YELLOW,
            Color.CYAN,
            Color.MAGENTA,
            Color.WHITE
        )

        for(potentialPrediction in rectsToTrack){
            val trackedPrediction = TrackedPrediction()
            trackedPrediction.detectionConfidence = potentialPrediction.first
            trackedPrediction.location = RectF(potentialPrediction.second.location)
            trackedPrediction.title = potentialPrediction.second.title

            trackedPrediction.color = colors[potentialPrediction.second.detectedClass % colors.size]
            trackedObjects.add(trackedPrediction)
        }

        for(lane in lanes){
            var trackedPrediction = TrackedPrediction()
            trackedPrediction.lanePoints = lane
            trackedPrediction.id = 1
            trackedObjects.add(trackedPrediction)
        }
    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun detectViolations(results: MutableList<Detector.Detection>, laneResults: MutableList<FloatArray>){
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
            var motorcycleList: MutableList<DetectorActivity.MotorcycleObject> = ArrayList()
            var helmetList: MutableList<Detector.Detection> = ArrayList()
            var lpList: MutableList<Detector.Detection> = ArrayList()
            var tricycleList: MutableList<Detector.Detection> = ArrayList()

            var index = 0
            //var helmetIndex = 0
            //var lpIndex = 0
            for(result in results){
                if(result.detectedClass == 0){                                  // detect if motorcycle
                    val motorcycleObject = DetectorActivity.MotorcycleObject()
                    result.id = index++.toString()                              // increment index identifier
                    motorcycleObject.motorcyclist = result                      // set the motorcyclist object
                    motorcycleObject.dateTime = FileUtil.dateToString(FileUtil.getDateTime())          // set datetime of violation
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

                        motorcycle.licensePlate = DetectorActivity.LicensePlate(lp)
                        motorcycle.snapshot = rgbFrameBitmap!!
                        DetectorActivity.readLp(motorcycle)
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

                    if(determinant > 0){
                        Log.d("PROC_VIOL", "COUNTERFLOW, PLACING IN POTENTIAL")
                        // counterflow detected
                        motorcycle.offenseCounterflow = "yes"
                    }
                }

                /**
                 * Check current potential violations list. If there is any, process the violations list.
                 */
                try{
                    val originalPotentialViolationsList = DetectorActivity.liveModel.getPotentialViolationsList() // Get potential list
                    var refTime = FileUtil.getDateTime()
                    for(potViolator in originalPotentialViolationsList!!){                      // check every pot vio
                        var potAddedToFinal = false
                        for(motorcycle in motorcycleList){
                            // check if lp of potential violator exists within detected motorcycle objects
                            // IF THIS IS TRUE, THEN CONFIRM THE VIOLATION.
                            if(potViolator.potentialViolation){
                                Log.d("TEST", "Succeed start")
                                if(potViolator.licensePlate != null){
                                    if(potViolator.motorcyclist!!.location.contains(motorcycle.licensePlate!!.detectedLicense.location)
                                        || potViolator.licensePlate!!.licenseNumber.equals(motorcycle.licensePlate!!.licenseNumber, true)
                                    ) {
                                        // If previous violator is still in this frame,
                                        // after 2 seconds, process it to final violation. Checking involves
                                        // looking at the relative locations of the motorcyclist, LP or a License Number match
                                        // and checking if they have the same data
                                        motorcycle.potentialViolation = true        // motorcycle has been found in pot vio list
                                        if ((FileUtil.elapsedTime(FileUtil.stringToDate(potViolator.dateTime), refTime) >= 2.0)) {
                                            // Remove from potential violations list
                                            Log.d("TEST", "Succeed end")
                                           // DetectorActivity.liveModel.removeFromPotentialViolationsList(potViolator)
                                            // Call and prepare to save data in device
                                            motorcycle.potentialViolation = false
                                            motorcycle.finalViolation = true
                                            //DetectorActivity.processViolation(potViolator)
                                            potAddedToFinal = true
                                            break
                                        }
                                    }
                                }else if((ModelUtils.box_iou(potViolator.motorcyclist!!.location, motorcycle.motorcyclist!!.location) < 0.9)){
                                    motorcycle.potentialViolation = true        // motorcycle has been found in pot vio list
                                    if ((FileUtil.elapsedTime(FileUtil.stringToDate(potViolator.dateTime), refTime) >= 2.0)) {
                                        // Remove from potential violations list
                                        Log.d("TEST", "nullSucceed end")
                                        //DetectorActivity.liveModel.removeFromPotentialViolationsList(potViolator)
                                        // Call and prepare to save data in device
                                        motorcycle.potentialViolation = false
                                        motorcycle.finalViolation = true
                                       // DetectorActivity.processViolation(potViolator)
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
                                DetectorActivity.liveModel.removeFromPotentialViolationsList(potViolator)
                                //Log.d("NO_VIOL", "No violators detected within current frame")
                            }
                        }
                    }
                }catch(e: Exception){
                    e.printStackTrace()
                }
                /**
                 * Place motorcycle into potential violation if non-existent yet. Next Frame, check if existing license plate is within potential
                 * violation list. (Code above)
                 */

                if(!motorcycle.potentialViolation && !motorcycle.finalViolation){

                    motorcycle.potentialViolation = true
                    if(motorcycle.helmet == null){
                        Log.d("PROC_VIOL", "HELMET NOT FOUND, PLACING IN POTENTIAL")
                        motorcycle.offenseHelmet = "yes"
                    }
                    // Add to potential Violations List
                    Log.d("PROC_VIOL", "Adding to potential Violations List")
                    DetectorActivity.liveModel.addToPotentialViolationsList(motorcycle)
                }
            }
        }catch(e: Exception){
            e.printStackTrace()
        }
    }

    fun initializeDetectors(){

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
        sensorOrientation = 0

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

        tracker.setFrameConfig(previewWidth, previewHeight, sensorOrientation)
    }

    fun setPreviewDimensions(width: Int, height: Int){
        previewWidth = width
        previewHeight = height
    }

    @Synchronized
    override fun onResume(){
        super.onResume()
        handlerThread = HandlerThread("inference")
        handlerThread!!.start()
        handler = Handler(handlerThread!!.looper)
    }

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

    @Synchronized
    override fun onStop() {
        super.onStop()
    }

    @Synchronized
    override fun onDestroy() {
        super.onDestroy()
    }

}