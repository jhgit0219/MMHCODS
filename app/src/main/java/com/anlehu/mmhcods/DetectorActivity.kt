package com.anlehu.mmhcods

import android.graphics.*
import android.media.ImageReader
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.util.TypedValue
import android.view.Surface
import android.view.WindowManager
import android.widget.Toast
import androidx.annotation.RequiresApi
import com.anlehu.mmhcods.utils.BorderedText
import com.anlehu.mmhcods.utils.Detector
import com.anlehu.mmhcods.utils.DetectorFactory
import com.anlehu.mmhcods.utils.ImageUtils
import com.anlehu.mmhcods.views.OverlayView
import java.lang.Exception
import java.util.*

class DetectorActivity: CameraActivity(), ImageReader.OnImageAvailableListener {

    private lateinit var trackingOverlay: OverlayView
    private lateinit var frameToCropMat: Matrix
    private lateinit var cropToFrameMat: Matrix
    private lateinit var borderedText: BorderedText
    private lateinit var tracker: BoxTracker
    private lateinit var detector: YoloV5Classifier

    private var sensorOrientation: Int = 0
    private var previewWidth = 0
    private var previewHeight = 0
    private var computingDetection: Boolean = false

    private var rgbFrameBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null
    private var copiedBitmap: Bitmap? = null

    private var DESIRED_PREVIEW_SIZE = Size(640, 640)

    private var MAINTAIN_ASPECT = true

    override fun getLayoutId(): Int {
        return R.layout.fragment_camera
    }

    override fun getDesiredPreviewFrameSize(): Size {
        return DESIRED_PREVIEW_SIZE
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
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

        //ImageUtils.saveBitmap(croppedBitmap!!, "preview.png")

        runInBackground {
            val results: List<Detector.Detection> = detector.detectImage(croppedBitmap!!)
            copiedBitmap = Bitmap.createBitmap(croppedBitmap!!)
            val canvas1 = Canvas(copiedBitmap!!)
            val paint = Paint()
            paint.color = Color.BLUE
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.0f

            val minConfidence = MainActivity.MINIMUM_CONFIDENCE

            var mappedPredictions: MutableList<Detector.Detection> = LinkedList()

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

            populateFrameDetections(mappedPredictions)

            computingDetection = false
        }
    }

    private fun populateFrameDetections(results: MutableList<Detector.Detection>) {
        /** Logic will be:
         * 1.) for result in results
         * 2.) if object is class motorcylist(0), create new object predictedMotorcyclist
         * 3.) since results is ordered by 0, 0, 1, 1, 2, 2, etc... then we can assume that the number of motorcyclist
         *      is what we are concerned with.
         * 4.) if object is class 1, create new helmetList and add this object
         * 5.) if object is class 2, create new license plate and add this object
         * 6.) Check each of their rects: If helmet rect is within motorcycle rect, add it to that predictedMotorycle
         *      object. Same with license plate
         * 7.) If rect is not, then discard
         * 8.) Finally, check each motorcycle object if their helmet has been initialized. Basically, if a motorcycle
         *      has a null helmet, then that motorcycle might not have a helmet. Add this to list of potential flags.
         *      Maybe add a license plate checker on the next frame to reconfirm if this object really does not have
         *      a helmet.
         */
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

    class MotorcycleObject(){
        var id: Int? = null
        var helmetState: Boolean = false
        var licensePlate: RectF? = null

        // Pass license plate recognition to a separate reader. Maintain id as identifier for which motorcycle owns which
        // license plate
    }
}
