package com.anlehu.mmhcods

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.RectF
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import com.anlehu.mmhcods.utils.DetectionComparator
import com.anlehu.mmhcods.utils.Detector
import com.anlehu.mmhcods.utils.ImageUtils
import com.anlehu.mmhcods.utils.ModelUtils
import com.anlehu.mmhcods.views.OverlayView
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskCompletionSource
import org.jetbrains.kotlinx.multik.api.arange
import org.jetbrains.kotlinx.multik.api.linspace
import org.jetbrains.kotlinx.multik.api.math.exp
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import org.jetbrains.kotlinx.multik.ndarray.operations.toMutableList
import org.opencv.android.Utils.matToBitmap
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.IOException
import java.lang.Boolean.FALSE
import java.lang.Boolean.TRUE
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.collections.set
import kotlin.math.pow


open class LaneClassifier()  : Detector {

    val tusimple_row_anchor = arrayOf( 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
        116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
        168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
        220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
        272, 276, 280, 284 )

    var isInitialized = false
        private set

    private var inputImageWidth: Int = 0 // will be inferred from TF Lite model.
    private var inputImageHeight: Int = 0 // will be inferred from TF Lite model.
    private var modelInputSize: Int = 0 // will be inferred from TF Lite model.

    private var outputImageWidth: Int = 0 // will be inferred from TF Lite model.
    private var outputImageHeight: Int = 0 // will be inferred from TF Lite model.
    private var modelOutputSize: Int = 0 // will be inferred from TF Lite model.

    private val executorService: ExecutorService = Executors.newCachedThreadPool()

    // Float model
    private val IMAGE_MEAN_R = 0.485f
    private val IMAGE_MEAN_G = 0.456f
    private val IMAGE_MEAN_B = 0.406f

    private val IMAGE_STD_R = 0.229f //mk.ndarray(mk[0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f])
    private val IMAGE_STD_G = 0.224f
    private val IMAGE_STD_B = 0.225f

    //config lane detector
    var INPUT_WIDTH: Int = -1
    var INPUT_HEIGHT: Int = -1

    private var output_box = 0

    private var isModelQuantized = false

    /** holds a gpu delegate  */
    var gpuDelegate: GpuDelegate? = null

    /** holds an nnapi delegate  */
    var nnapiDelegate: NnApiDelegate? = null

    /** The loaded TensorFlow Lite model.  */
    private var tfliteModel: MappedByteBuffer? = null

    /** Options for configuring the Interpreter.  */
    private val tfliteOptions: Interpreter.Options = Interpreter.Options()

    // Add / replace variables here
    private lateinit var trackingOverlay: OverlayView

    // Config values.
    // Pre-allocated buffers.
    private var lane = object {
        var recentFit:MutableList<Array<FloatArray>> = ArrayList()
        var avgFit: List<FloatArray> = ArrayList()
    }

    private val labels = Vector<String>()
    private var intValues: IntArray = IntArray(288 * 800)

    private var imgData: ByteBuffer? = null
    private var outData: ByteBuffer? = null

    private var tfLite: Interpreter? = null
    private var inp_scale = 0f
    private var inp_zero_point = 0
    private var oup_scale = 0f
    private var oup_zero_point = 0
    private var numClass = 0

//
//    fun imageToBitmap() {
////        val d : Drawable
////        d = context.resources.getDrawable(R.drawable.detected_lanes)
////        val bitmap = Bitmap.createBitmap(
////            1280,
////            720,
////            Bitmap.Config.ARGB_8888)
//
//        //val path : String = getExternalStorageDirectory().getAbsolutePath() + "/detected_lanes.png"
//        var imgMat = Imgcodecs.imread("/storage/emulated/0/Download/moto.jpg")
//        Log.d("IMG_MAT_U", imgMat.get(0, 0).toString())
//        //imgMat.convertTo(imgMat, CvType.CV_32FC4)
//        Imgproc.cvtColor(imgMat, imgMat, Imgproc.COLOR_BGR2RGB)
//        Log.d("IMG_MAT_C", imgMat.get(0, 0).toString())
//
////        val tempBitmap : Bitmap
////        org.opencv.android.Utils.matToBitmap(imgMat,tempBitmap)
//        //val bitmap = BitmapFactory.decodeResource(context.resources, R.drawable.detected_lanes)
//        Log.d("Image","Mat: "+imgMat.toString())
////        Log.i("bitmap",$bitmap)
//        classifyAsync(imgMat)
//
//    }

    fun classify(mat: Mat): String {
        var resMat : Mat = Mat()
        mat.copyTo(resMat)
        // TODO: Add code to run inference with TF Lite.
        // Pre-processing: resize the input image to match the model input shape.

//        var bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888)
//        matToBitmap(mat,bitmap)
//        var resizedImage = Bitmap.createScaledBitmap(
//            bitmap,
//            inputImageWidth,
//            inputImageHeight,
//            true
//        )
        val imgSize = Size(800.0, 288.0)
        Imgproc.resize(resMat, resMat, imgSize, 0.0, 0.0, Imgproc.INTER_LINEAR)    // Resizes image
        // Converts image to tensor not needed since this is in RGB already

        //Log.d("ResizedImage","resized image: "+resizedImage.toString())
        //resizedImage = resizedImage(DataType.FLOAT32)X

        // mat.convertTo(mat,CvType.CV_32FC3)
        // image normalization process

        //val bytes = ByteArray(mat.rows() * mat.cols() * mat.channels()*4)
        //mat.get(0,0,bytes)

        // val resbytes = ByteArray(resMat.rows() * resMat.cols() * resMat.channels()*4)
        //resMat.get(0,0,resbytes)

        //var byteBuffer = ByteBuffer.wrap(bytes)
        //var byteBuffer = FloatBuffer.wrap(bytes)

//        // TEST
//        val buffer = byteBuffer.asFloatBuffer()
//        buffer.rewind()
//        val bitmap2 = Bitmap.createBitmap(800, 288, Bitmap.Config.ARGB_8888)
//        bitmap2.copyPixelsFromBuffer(buffer)
//        ImageUtils.saveBitmap(bitmap2, "COpied.png")

        var bitmap = Bitmap.createBitmap(1280, 720, Bitmap.Config.ARGB_8888)
        var resBitmap = Bitmap.createBitmap(800, 288, Bitmap.Config.ARGB_8888)
        matToBitmap(resMat,resBitmap)
        Log.d("ASSERT", "${mat.dims()} ${mat.width()} ${mat.height()}")
        matToBitmap(mat, bitmap)
        Log.d("SIZE_H", bitmap.height.toString())
        var byteBuffer = convertBitmapToByteBuffer(resBitmap) // This turns 4 channel bitmap to 3-channel bytebuffer
//        var tensorImageType = tfLite!!.getInputTensor(0).dataType()
//        val tensorImage = TensorImage(tensorImageType)
//        tensorImage.load(bitmap)
//
//        finBitmap = bitmap
//        /** Create Image Processor **/
//        val imageProcessor = ImageProcessor.Builder()
//            .add(ResizeOp(288, 800, ResizeOp.ResizeMethod.BILINEAR))
//            .add(ResizeWithCropOrPadOp(288, 800)).build()
//
//        val procImage = imageProcessor.process(tensorImage)
//
//        Log.d("SAVING_TENSOR", "TO BITMAP")
//        // TEST
//        val buffer = procImage.buffer
//        buffer.rewind()
//        val bitmap2 = Bitmap.createBitmap(800, 288, Bitmap.Config.ARGB_8888)
//        bitmap2.copyPixelsFromBuffer(buffer)
//        ImageUtils.saveBitmap(bitmap2, "Tensor.png")
//        Log.d("SAVING_TENSOR", "SUCCESS")
//
//        val imageTensorBuffer = procImage.tensorBuffer
//        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
//        val stddev = floatArrayOf(0.229f, 0.224f, 0.225f)
//        val tensorProcessor = TensorProcessor.Builder().add(NormalizeOp(mean, stddev)).add(CastOp(DataType.FLOAT32)).build()
//        val processedTensor = tensorProcessor.process(imageTensorBuffer)
        //Log.d("bytebuffer","bytebuffer: "+byteBuffer.toString())
        //byteBuffer = byteBuffer(DataType.FLOAT32)
        //var floatByteBuffer = byteBuffer(DataType.FLOAT32)

        //val tensorBuffer = TensorBuffer.createFixedSize(byteBuffer, DataType.FLOAT32);

        // Define an array to store the model output.
        val output = Array(1) {
            Array(101) {
                Array(56) {
                    FloatArray(4)
                }
            }
        }
        Log.d("output r","output row: "+output.size.toString())
        Log.d("output c","output column: "+output[0].size.toString())
        // Run inference with the input data.
        //byteBuffer!!.rewind()
        tfLite!!.run(byteBuffer, output)
        //tfLite!!.run(processedTensor.buffer, output)
        // Post-processing: find the digit that has the highest probability
        // and return it a human-readable string.


        val result = output[0]          //Checking if this is the output to use (output[0]) or just output

        /* val resized_Result = Bitmap.createScaledBitmap(
             output,
             outputImageWidth,
             outputImageHeight,
             true

         )*/
        Log.d("Result output","result: "+result.contentDeepToString())
//        val maxIndex = result.indices.maxByOrNull { result[it] } ?: -1
//        val resultString =
//            "Prediction Result: %d\nConfidence: %2f"
//                .format(maxIndex, result[maxIndex])
//
//        Log.d("ResultString","resultString: "+resultString.toString())
//        return resultString
        processOutput(output)
        return "nice"
    }

    fun classifyAsync(mat: Mat): Task<String> {
        val task = TaskCompletionSource<String>()
        executorService.execute {
            val result = classify(mat)
            task.setResult(result)
        }
        return task.task
    }

    override fun detectImage(bitmap: Bitmap): ArrayList<Detector.Detection> {
        // resize bitmap from camera
        //var resbitmap = Bitmap.createScaledBitmap(bitmap, 800, 288, true)
        //bitmap is already resized; normalize now
        ImageUtils.saveBitmap(bitmap, "lane")
        convertBitmapToByteBuffer(bitmap)

        // output array
        val output = Array(1) {
            Array(101) {
                Array(56) {
                    FloatArray(4)
                }
            }
        }

        Log.d("output r","output row: "+output.size.toString())
        Log.d("output c","output column: "+output[0].size.toString())
        // Run inference with the input data.
        //byteBuffer!!.rewind()
        tfLite!!.run(imgData, output)
        //tfLite!!.run(processedTensor.buffer, output)
        // Post-processing: find the digit that has the highest probability
        // and return it a human-readable string.


        val result = output[0]          //Checking if this is the output to use (output[0]) or just output

        /* val resized_Result = Bitmap.createScaledBitmap(
             output,
             outputImageWidth,
             outputImageHeight,
             true

         )*/
        Log.d("Result output","result: "+result.contentDeepToString())
//        val maxIndex = result.indices.maxByOrNull { result[it] } ?: -1
//        val resultString =
//            "Prediction Result: %d\nConfidence: %2f"
//                .format(maxIndex, result[maxIndex])
//
//        Log.d("ResultString","resultString: "+resultString.toString())
//        return resultString
        processOutput(output)

        // create output array

        return ArrayList()
    }

    override fun close() {
        executorService.execute {
            tfLite?.close()
            Log.d(TAG, "Closed TFLite interpreter.")
        }
    }

    override fun setNumOfThreads(numThread: Int) {
        if (tfLite != null) tfliteOptions.setNumThreads(numThread)
    }

    override fun setUseNNAPI(isChecked: Boolean) {
//        if (tfLite != null) tfLite.setUseNNAPI(isChecked);
    }

    private fun recreateInterpreter() {
        if (tfLite != null) {
            tfLite!!.close()
            tfLite = Interpreter(tfliteModel as ByteBuffer, tfliteOptions)
        }
    }

    fun useGpu() {
        if (gpuDelegate == null) {
            gpuDelegate = GpuDelegate()
            tfliteOptions.addDelegate(gpuDelegate)
            recreateInterpreter()
        }
    }

    fun useCPU() {
        recreateInterpreter()
    }

    fun useNNAPI() {
        nnapiDelegate = NnApiDelegate()
        tfliteOptions.addDelegate(nnapiDelegate)
        recreateInterpreter()
    }

    override fun getObjThresh(): Float {
        return MainActivity.MINIMUM_CONFIDENCE
    }

    //non maximum suppression
    @RequiresApi(Build.VERSION_CODES.N)
    open fun nms(list: ArrayList<Detector.Detection>): ArrayList<Detector.Detection> {
        val nmsList = ArrayList<Detector.Detection>()
        for (k in labels.indices) {
            //1.find max confidence per class
            val pq: PriorityQueue<Detector.Detection> = PriorityQueue<Detector.Detection>(
                DetectionComparator
            )
            for (i in list.indices) {
                if (list[i].detectedClass == k) {
                    pq.add(list[i])
                }
            }

            //2.do non maximum suppression
            while (pq.size > 0) {
                //insert detection with max confidence
                val a: Array<Detector.Detection?> = arrayOfNulls(pq.size)
                val detections: Array<Detector.Detection> = pq.toArray(a)
                val max: Detector.Detection = detections[0]
                nmsList.add(max)
                pq.clear()
                for (j in 1 until detections.size) {
                    val detection: Detector.Detection = detections[j]
                    val b: RectF = detection.location
                    if (box_iou(max.location, b) < mNmsThresh) {
                        pq.add(detection)
                    }
                }
            }
        }
        return nmsList
    }


    private val mNmsThresh = 0.6f

    fun box_iou(a: RectF, b: RectF): Float {
        return box_intersection(a, b) / box_union(a, b)
    }

    fun box_intersection(a: RectF, b: RectF): Float {
        val w: Float = overlap(
            (a.left + a.right) / 2, a.right - a.left,
            (b.left + b.right) / 2, b.right - b.left
        )
        val h: Float = overlap(
            (a.top + a.bottom) / 2, a.bottom - a.top,
            (b.top + b.bottom) / 2, b.bottom - b.top
        )
        return if (w < 0.0 || h < 0.0) 0f else (w * h)
    }

    fun box_union(a: RectF, b: RectF): Float {
        val i = box_intersection(a, b)
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
    }

    fun overlap(x1: Float, w1: Float, x2: Float, w2: Float): Float {
        val l1 = x1 - w1 / 2
        val l2 = x2 - w2 / 2
        val left = if (l1 > l2) l1 else l2
        val r1 = x1 + w1 / 2
        val r2 = x2 + w2 / 2
        val right = if (r1 < r2) r1 else r2
        return right - left
    }

    protected val BATCH_SIZE = 1
    protected val PIXEL_SIZE = 3


    // Writes Image data into a `ByteBuffer`.

    private fun convertBitmapToByteBuffer(bitmap: Bitmap?){
//        val byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * 288 * 800 * 3)
//        byteBuffer.order(ByteOrder.nativeOrder())
//        val intValues = IntArray(288*800)
        bitmap!!.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val pixel = 0
        imgData!!.rewind()
        for (i in 0 until INPUT_HEIGHT) {
            for (j in 0 until INPUT_WIDTH) {
                val pixelValue = intValues[i * INPUT_WIDTH + j]
                if (isModelQuantized) {/*
                    // Quantized model
                    byteBuffer.put((((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point) as Byte)
                    byteBuffer.put((((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point) as Byte)
                    byteBuffer.put((((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point) as Byte)*/
                } else { // Float model

                    imgData!!.putFloat((((pixelValue shr 16 and 0xFF)/255.0f) - IMAGE_MEAN_R) / IMAGE_STD_R)
                    imgData!!.putFloat((((pixelValue shr 8 and 0xFF)/255.0f) - IMAGE_MEAN_G) / IMAGE_STD_G)
                    imgData!!.putFloat((((pixelValue and 0xFF)/255.0f) - IMAGE_MEAN_B) / IMAGE_STD_B)

                }
            }
        }
        // TEST
        val buffer = imgData
        buffer!!.rewind()
        val bitmap2 = Bitmap.createBitmap(800, 288, Bitmap.Config.ARGB_8888)
        bitmap2.copyPixelsFromBuffer(buffer)
        ImageUtils.saveBitmap(bitmap2, "Copied")
    }

    //fun detectLane(bitmap: Bitmap): Array<FloatArray> {

    @RequiresApi(Build.VERSION_CODES.N)
    fun detectLane(bitmap: Bitmap): FloatArray {
        //convertBitmapToByteBuffer(bitmap)

        val outputMap: MutableMap<Int?, Any?> = HashMap()

        outData!!.rewind()
        Log.d("DetectImage", "$output_box")
        outputMap[0] = outData
        val inputArray: Array<out Any> = arrayOf(imgData as Any)
        tfLite!!.runForMultipleInputsOutputs(inputArray, outputMap)

        val byteBuffer = outputMap[0] as ByteBuffer
        byteBuffer.rewind()

        // grab prediction
        val out =
            Array(80) {
                FloatArray(160)
            }

        for(i in 0 until 80){
            for(j in 0 until 160){
                // since byte buffer is a flat buffer, the values are arranged of pixels per row
                // thus, to iterate per row, simply multiply i by 80
                val pixelValue = byteBuffer[i * 160 + j]
                // un-normalize the value by multiplying by 255 (RGB)
                if ((pixelValue.toFloat() * 255f) < 1)
                    out[i][j] = 0f
                else
                    out[i][j] = pixelValue.toFloat() * 255f
            }
        }
        // Add this prediction to lane object
        lane.recentFit.add(out)
        // Only use the last 5 predictions
        if(lane.recentFit.size > 5){
            val list: MutableList<Array<FloatArray>> = ArrayList()
            for(i in 1..5){
                list.add(lane.recentFit[i])
            }
            lane.recentFit = list
        }

        // Calculate the average detection per pixel
        // For every row
        for(i in 0 until 80){
            // For every column
            for(j in 0 until 160){
                // For every index
                var average = 0f
                for(x in 0 until lane.recentFit.size){
                    average += lane.recentFit[x][i][j]
                }
                average / lane.recentFit.size.toFloat() // mean of pixel
                out[i][j] = average
            }
        }
        var array= ArrayList<Float>()
        for(i in 0 until 80){
            var count = 0
            for(j in 0 until 160){
                if(out[i][j] > 1 && count < 5){
                    //Log.d("LaneClassifier", "LANE DETECTED")
                    // add x point
                    array.add(j.toFloat())
                    array.add(79f-i)
                    ++count
                }
            }
        }

        lane.avgFit = out.toList()
        /**
         *  The "stride" value is equal to the height value because this is a flat array
         *  e.g. Each height (i) contains a value from index 0 to width. Traversing to the next i,
         *  just have to multiply i by the width ( next i, j=0)
         */

        return array.toFloatArray()
    }


    fun processOutput(imgData: Array<Array<Array<FloatArray>>>)
    {

        val processed_output = mutableListOf<MutableList<MutableList<Float>>>()

        for(i in 0..100){
            processed_output.add(mutableListOf())  // Add a mutableList at index i
            for(j in 55 downTo 0){
                processed_output[i].add(imgData[0][i][j].toMutableList()) // add the last element of the 2nd dimension
                // as the 1st element of every 1st dimension
                // list
            }
        }

        var processed_output_ndarry = mk.ndarray(processed_output)

        Log.d("Processed Output","processed_output: "+processed_output.toString())
        Log.d("Processed Output NDarry","processed_output_ndarry: "+processed_output_ndarry.toString())

        //(processed_output[:-1, :, :]

        var prob_processed_output = mutableListOf<MutableList<MutableList<Float>>>()

        for(i in 0..99)
        {
            prob_processed_output.add(mutableListOf())
            for(j in 0 .. 55)
            {
                prob_processed_output[i].add(imgData[0][i][j].toMutableList())
            }
        }

        var prob_processed_output_ndarry = mk.ndarray(prob_processed_output)

//        Log.d("Prob Processed Output","prob_processed_output: "+prob_processed_output.toString())
//        Log.d("Prob Processed Output NDarry","prob_processed_output_ndarry: "+prob_processed_output_ndarry.toString())

        //var prob = softmax(prob_processed_output, 0)   //convert to kotlin
        //can also try this for softmax
        var prob_e = mk.math.exp(prob_processed_output_ndarry)

//        Log.d("Prob Exp","prob_e: "+prob_e.toString())

        //var prob = prob_e / mk.math.sum(prob_e , axis = 0)      //var prob = prob_e / prob_e.sum() // or
        var prob = prob_e / mk.math.sum(prob_e)

        Log.d("Prob","prob: "+prob.toString())

        val griding_num = mk.arange<Int>(1,101,1)                           // create array 1-100

//        Log.d("Griding Number","griding_num: "+griding_num.toString())

        val idx = griding_num.reshape(griding_num.shape[0],1,1)
        Log.d("IDX","idx: "+idx.toString())


        var loc_prob_idx = mutableListOf<MutableList<MutableList<Float>>>()
        for(i in 0 until prob.shape[0])
        {
            loc_prob_idx.add(mutableListOf())
            for (j in 0 until prob.shape[1])
            {
                loc_prob_idx[i].add(mutableListOf())

                for(k in 0 until prob.shape[2])
                {
                    loc_prob_idx[i][j].add((prob[i][j][k] * idx[i][0][0]))
                }
            }
        }

        var loc_prob_idx_ndarray = mk.ndarray(loc_prob_idx)


//        Log.d("Loc_prob_idx","loc_prob_idx: "+loc_prob_idx.toString())
//        Log.d("Loc_prob_idx_ndarray","loc_prob_idx_ndarray: "+loc_prob_idx_ndarray.toString())
        var loc = mk.math.sumD3(loc_prob_idx_ndarray, axis = 0)                  //either this or the one below
        //loc = mk.math.cumSum(loc_prob_idx, axis = 0)


        Log.d("Loc","loc: "+loc.toString())

        //processed_output_ndarry = mk.math.argMax(processed_output_ndarry.asD3Array(), axis = 0)
        var argMax_process_output = mk.math.argMaxD3(processed_output_ndarry.asD3Array(), axis = 0)

        Log.d("argMax","argMax: "+argMax_process_output.toString())

        var loc_Argmax = mutableListOf<MutableList<MutableList<Float>>>()

        for(i in 0..(argMax_process_output.shape[0]-1))
        {
            loc_Argmax.add(mutableListOf())
            for(j in 0..(argMax_process_output.shape[1]-1))
            {
                loc_Argmax[i].add(mutableListOf())
                if(argMax_process_output[i][j].toFloat() != 100F)
                {
                    loc_Argmax[i][j].add(argMax_process_output[i][j].toFloat())
                }
                else
                {
                    loc_Argmax[i][j].add(0F)
                }
            }
        }

        var loc_Argmax_ndarray = mk.ndarray(loc_Argmax)

        val loc_processed_output = loc_Argmax_ndarray

//        Log.d("Loc Processed Output","loc_processed_output: "+loc_processed_output.toString())

        val col_sample = mk.linspace<Float>(0, 800-1, 100)

//        Log.d("Col Sample","col_sample: "+col_sample.toString())

        val col_sample_w = col_sample[1] - col_sample[0]

//        Log.d("Col Sample W","col_sample_w: "+col_sample_w.toString())

        var max_lanes = loc_processed_output.shape[1]                //convert to kotlin (get processed_output columns)
        //temporary
        Log.d("Max Lanes","max_lanes: "+max_lanes.toString())
        //val max_lanes = processed_output.length()
        //val lane_num : Float

        val lane_points_mat = ArrayList<ArrayList<ArrayList<Int>>>()
        val lanes_detected = ArrayList<Boolean>()

        for (lane_num in 0..(max_lanes-1))
        {
            var lane_points = ArrayList<ArrayList<Int>>()

            var sumLocPO = mutableListOf<MutableList<Float>>()
            for(i in 0..(loc_processed_output.shape[0]-1))
            {
                sumLocPO.add(mutableListOf())
                sumLocPO[i] = loc_processed_output[i][lane_num].toMutableList()
                // sumLocPO[i].add(loc_processed_output[0][i][lane_num])

//                Log.d("loc_processed_output[0][i][lane_num]",
//                    "loc_processed_output[0][i][lane_num]: "
//                            +loc_processed_output[0][i][lane_num].toString())
            }

            var sumLocPO_ndarray = mk.ndarray(sumLocPO)
//            Log.d("Ndarray for Loc_PO sum","sumLocPO_ndarray: "+sumLocPO_ndarray.toString())
//            Log.d("Loc_proc_out","loc_proc_out: "+loc_processed_output.shape[0].toString())
//            Log.d("Loc_proc_out","loc_proc_out: "+loc_processed_output.shape[1].toString())
//            Log.d("Loc_proc_out","loc_proc_out: "+loc_processed_output.shape[2].toString())
            //if (mk.math.sum(loc_processed_output[:,lane_num]!= 0)>2) //convert to kotlin

            if (mk.math.sum(sumLocPO_ndarray)>2)
            {
                lanes_detected.add(TRUE)
                for(point_num in 0..(loc_processed_output.shape[0]-1))             //convert to kotlin (get processed_output rows) -> processed_output[0].length()
                {
                    var thispoint = loc_processed_output[point_num,lane_num]
                    // Log.d("Loc Processed Output","loc_processed_output: "+loc_processed_output[point_num][lane_num][0].toString())
                    if (loc_processed_output[point_num][lane_num][0] > 0F) {
                        var lane_point_left = (loc_processed_output[point_num,lane_num,0] * col_sample_w * 1200 / 800).toInt() -1
                        //Log.d("Lane point left","lane_point_left: "+lane_point_left.toString())


//                        Log.d("tusimple_row_anchor1","tusimple_row_anchor1: "+(tusimple_row_anchor[55-point_num]/288).toString())
//                        Log.d("tusimple_row_anchor2","tusimple_row_anchor2: "+((tusimple_row_anchor[55-point_num]/288)*720).toString())
                        var lane_point_right = (720 * (tusimple_row_anchor[55-point_num].toFloat() / 288)).toInt() - 1
                       // Log.d("Lane point right","lane_point_right: "+lane_point_right.toString())


                        var lane_point = arrayListOf(lane_point_left,lane_point_right)
                       // Log.d("Lane point","lane_point: "+lane_point.toString())
                        lane_points.add(lane_point)
                    }
                }
            }
            else
            {
                lanes_detected.add(FALSE)
            }

            lane_points_mat.add(lane_points)
        }

        Log.d("Lane Detected","Lane_detected: "+lanes_detected.toString())

        if(lanes_detected[0])
        {
            Log.d("Lane POINTS Lane0","Lane_points_mat: "+lane_points_mat[0].toString())
        }
        if(lanes_detected[1])
        {
            Log.d("Lane POINTS Lane1","Lane_points_mat: "+lane_points_mat[1].toString())
        }
        if(lanes_detected[2])
        {
            Log.d("Lane POINTS Lane2","Lane_points_mat: "+lane_points_mat[2].toString())
        }
        if(lanes_detected[3])
        {
            Log.d("Lane POINTS Lane3","Lane_points_mat: "+lane_points_mat[3].toString())
        }

        /** Return a list with lanes with bestFit **/

        /** ADD LANE DRAWING **/
        val tempBitmap = Bitmap.createBitmap(1280, 720, Bitmap.Config.ARGB_8888)
        val tempCanvas = Canvas(tempBitmap)

        // draw on the canvas
        //tempCanvas.drawBitmap(bitmap, 0f, 0f, null)

        val lane_points = lane_points_mat[1].flatten().toList()
        val lane_points2 = lane_points_mat[2].flatten().toList()
        val points = FloatArray(lane_points.size)
        val points2 = FloatArray(lane_points2.size)

        for (i in lane_points.indices){
            points[i] = lane_points[i].toFloat()
        }
        for (i in lane_points2.indices){
            points2[i] = lane_points2[i].toFloat()
        }
        var fitPoint1 : FloatArray? = null
        var fitPoint2 : FloatArray? = null
        // best fit of line points
        if(points.size >= 4 )
            fitPoint1 = bestFit(points)
        if(points2.size >= 4 )
            fitPoint2 = bestFit(points2)

        Log.d("POINTS", "${points.joinToString(" ")}")
        // create lanes

//        val imageView = TestImageView(this.context)
//        val linearLayout = LinearLayout(this.context)
//        linearLayout.orientation = LinearLayout.VERTICAL
//        linearLayout.layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
//        imageView.setBackgroundColor(Color.BLACK)
//        imageView.layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
//        //imageView.setImageResource(R.drawable.detected_lanes)
//
//        val act = this.context as Activity
//        imageView.setImageDrawable(BitmapDrawable(act.resources, tempBitmap))
//
//        linearLayout.addView(imageView)
//        linearLayout.visibility = LinearLayout.VISIBLE
//
//        //layoutInflater.inflate(R.layout.activity_main, linearLayout, false)
//
//        runOnUiThread {
//            act.addContentView(linearLayout, linearLayout.layoutParams)
//            //imageView.draw(tempCanvas)
//            imageView.drawLane(fitPoint1, fitPoint2)
//            imageView.invalidate()
//            //act.setContentView(R.layout.activity_main)
//        }

    }

    fun bestFit(points: FloatArray): FloatArray{
        val length = points.size / 2
        var m = 0f
        var b = 0f
        var sum_x = 0f
        var sum_y = 0f
        var sum_xy = 0f
        var sum_x2 = 0f

        for (i in points.indices step 2){
            sum_x += points[i]
            sum_y += points[i+1] // negative form of the y
            sum_xy += points[i] * points[i+1]
            sum_x2 += points[i].toDouble().pow(2.0).toFloat()
        }

        m = (length * sum_xy - sum_x * sum_y) / (length * sum_x2 - sum_x.toDouble().pow(2.0).toFloat())
        b = (sum_y - m * sum_x) / length

        Log.d("SLOPE", "Slope: $m and intercept $b")

        //create Float Array of points
        // In this case, we want to get the x-values based on the y-values
        val bestFitArr = FloatArray(4)
        //insert first x using equation x = (y - b) / m
        bestFitArr[0] = ((points[1] - b) / m)-20f
        //insert y at first x using equation mx + b
        bestFitArr[1] = points[1]
        bestFitArr[3] = points[points.size-1]// last index is the last y-value, 2nd to the last is the last x-value
        //insert last x
        bestFitArr[2] = ((bestFitArr[3] - b) / m) - 50f


//        //determine if line is going to the left or right (from top to bottom)
//        if(bestFitArr[0] > bestFitArr[2]) { // if x0 is greater than x1
//            //adjust x0
//            Log.d("LANE_DIR","Line going from bot to top right")
//            bestFitArr[0] - 50f
//        }else{
//            //adjust x1
//            Log.d("LANE_DIR","Line going from top to bot right")
//            bestFitArr[0] - 50f
//        }

        return bestFitArr
    }

    companion object {

        private const val TAG = "LaneClassifier"

        private const val FLOAT_TYPE_SIZE = 4
        private const val PIXEL_SIZE = 1

        private const val OUTPUT_CLASSES_COUNT = 10
        // Number of threads in the java app

        val NUM_THREADS = 2
        val isNNAPI = false
        val isGPU = false

        var mountedPath = ""

        /**
         * Initializes a native TensorFlow session for classifying images.
         *
         * @param assetManager  The asset manager to be used to load assets.
         * @param modelFilename The filepath of the model GraphDef protocol buffer.
         * @param labelFilename The filepath of label file for classes.
         * @param isQuantized   Boolean representing model is quantized or not
         */
        @Throws(IOException::class)
        fun create(
            mountedPath: String,
            modelFilename: String?,
            isQuantized: Boolean,
            inputWidth: Int,
            inputHeight: Int
        ): LaneClassifier {
            val d = LaneClassifier()
            Log.d("FileName: ", "$modelFilename")
            try {
                val options: Interpreter.Options = Interpreter.Options()
                options.setNumThreads(NUM_THREADS)
                if (isNNAPI) {
                    d.nnapiDelegate = null
                    // Initialize interpreter with NNAPI delegate for Android Pie or above
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                        d.nnapiDelegate = NnApiDelegate()
                        options.addDelegate(d.nnapiDelegate)
                        options.setNumThreads(NUM_THREADS)
                        //                    options.setUseNNAPI(false);
//                    options.setAllowFp16PrecisionForFp32(true);
//                    options.setAllowBufferHandleOutput(true);
                        options.setUseNNAPI(true)
                    }
                }
                if (isGPU) {
                    //val gpu_options: GpuDelegate.Options = GpuDelegate.Options()
                    //gpu_options.setPrecisionLossAllowed(true)
                    //gpu_options.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                    // d.gpuDelegate = GpuDelegate(gpu_options)
                    //options.addDelegate(d.gpuDelegate)
                    options.apply{
                        this.addDelegate(GpuDelegate(CompatibilityList().bestOptionsForThisDevice))
                    }
                }
                d.tfliteModel = ModelUtils.loadModelFile(mountedPath,modelFilename)
                d.tfLite = Interpreter(d.tfliteModel!!, options)
                Log.d("Lane_GPU","Creating lane model")
            } catch (e: Exception) {
                throw RuntimeException(e)
            }
            d.isModelQuantized = isQuantized
            Log.d("LaneClassifier", "Quantized: $isQuantized")
            // Pre-allocate buffers.
            val numBytesPerChannel: Int = if (isQuantized) {
                1 // Quantized
            } else {
                4 // Floating point
            }
            d.INPUT_HEIGHT = inputHeight
            d.INPUT_WIDTH = inputWidth
            d.imgData = ByteBuffer.allocateDirect(1 * d.INPUT_WIDTH * d.INPUT_HEIGHT * 3 * numBytesPerChannel)
            d.imgData!!.order(ByteOrder.nativeOrder())
            d.intValues = IntArray(d.INPUT_WIDTH * d.INPUT_HEIGHT)
            /*d.output_box =
                (((inputSize / 32.0f).pow(2.0f) + (inputSize / 16.0f).pow(2.0f) + (inputSize / 8.0f).pow(2.0f)) * 3).toInt()*/
            Log.d("ImgData Lane", "${d.INPUT_WIDTH} x ${d.INPUT_HEIGHT}")
            if (d.isModelQuantized) {
                val inpten: Tensor = d.tfLite!!.getInputTensor(0)
                d.inp_scale = inpten.quantizationParams().scale
                d.inp_zero_point = inpten.quantizationParams().zeroPoint
                val oupten: Tensor = d.tfLite!!.getOutputTensor(0)
                d.oup_scale = oupten.quantizationParams().scale
                d.oup_zero_point = oupten.quantizationParams().zeroPoint
            }
            val shape: IntArray = d.tfLite!!.getOutputTensor(0).shape()
            val numClass = shape[shape.size - 1]
            Log.d("NumClass", "$numClass")
            d.numClass = numClass
            d.outData = ByteBuffer.allocateDirect(d.output_box * numBytesPerChannel)
            d.outData!!.order(ByteOrder.nativeOrder())
            return d
        }

    }
}