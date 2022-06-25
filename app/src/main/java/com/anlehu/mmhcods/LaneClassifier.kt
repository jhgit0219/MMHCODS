package com.anlehu.mmhcods

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Build
import android.os.storage.OnObbStateChangeListener
import android.os.storage.StorageManager
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.anlehu.mmhcods.utils.DetectionComparator
import com.anlehu.mmhcods.utils.Detector
import com.anlehu.mmhcods.utils.Utils
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
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.lang.Boolean.FALSE
import java.lang.Boolean.TRUE
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.collections.List
import kotlin.collections.MutableList
import kotlin.collections.MutableMap
import kotlin.collections.arrayListOf
import kotlin.collections.contentDeepToString
import kotlin.collections.indices
import kotlin.collections.mutableListOf
import kotlin.collections.set
import kotlin.collections.toFloatArray
import kotlin.collections.toList
import kotlin.collections.toMutableList


class LaneClassifier(context: Context)  : AppCompatActivity(),  Detector {



    val tusimple_row_anchor = arrayOf( 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
        116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
        168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
        220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
        272, 276, 280, 284 )
    private var interpreter: Interpreter? = null
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
    private val IMAGE_MEAN = 0f

    private val IMAGE_STD = 255.0f

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

    var lane_points_mat = ArrayList<ArrayList<ArrayList<Int>>>()
    var lanes_detected = ArrayList<Boolean>()

    // Add / replace variables here
    private lateinit var trackingOverlay: OverlayView
    val context = context
    val obbPath = this.context.obbDir.absolutePath
    var mountedPath = ""
    //val storageManager = StorageManager().mountObb(obbPath)
    val storageManager = context.getSystemService(Context.STORAGE_SERVICE) as StorageManager

    val mListener = object : OnObbStateChangeListener() {
        override fun onObbStateChange(path: String, state: Int) {
            Log.d("HELLO_W", "TEST VALUE $state")
            if(state == MOUNTED){
                Log.d("MOUNTED:", "${storageManager.getMountedObbPath(path)}")
                mountedPath = storageManager.getMountedObbPath(path)
            }else if (state == ERROR_ALREADY_MOUNTED){
                Log.d("ALREADY MOUNTED:", path)
            }else{
                Log.d("ERROR:", path)
            }
        }
    }


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

    //for imageToBitmap (can delete once not needed anymore)


//    public fun LaneClassifier()
//    {
//    }
    init{

        val x = storageManager.unmountObb(obbPath+"/main.1.com.anlehu.mmhcods.obb", true, mListener)
        Log.d("UNMOUNT: ", "$x")
        val y = storageManager.mountObb(obbPath+"/main.1.com.anlehu.mmhcods.obb", null, mListener)
        Log.d("MOUNT: ", "$y")

        try{
            this.tfliteOptions.apply{
                this.addDelegate(GpuDelegate(CompatibilityList().bestOptionsForThisDevice))
            }
        }catch(e: Exception){
            Log.e("TFLITE_ERROR", e.toString())
        }


    }


    fun initialize(): Task<Void?> {

        val task = TaskCompletionSource<Void?>()
        executorService.execute {
            try {
                initializeInterpreter()
                imageToBitmap()
                task.setResult(null)
            } catch (e: IOException) {
                task.setException(e)
            }
        }

        return task.task
    }


    fun initializeInterpreter(){

        //val model = loadModelFile(assetManager, "model_float32.tflite")
        //val model = loadModelFile("model_float32.tflite")
        //this.interpreter = loadModelFileLite("model_float32.tflite")

        //val interpreter = Interpreter(model)
        Log.d("tflite", "${mountedPath}/model_float32.tflite")
        this.tfliteModel = Utils.loadModelFile(this.context.assets,"${mountedPath}/model_float32.tflite")
        this.tfLite = Interpreter(tfliteModel!!, tfliteOptions)
        // Finish interpreter initialization.
        isInitialized = true
        Log.d(TAG, "Initialized TFLite interpreter.")

        val inputShape = tfLite!!.getInputTensor(0).shape()
        inputImageWidth = inputShape[1]
        inputImageHeight = inputShape[2]
        Log.d("INTPUTIMAGEWIDTH","INTPUTIMAGEWIDTH"+inputImageWidth.toString())
        Log.d("INPUTIMAGEHEIGHT","INPUTIMAGEHEIGHT"+inputImageHeight.toString())
        modelInputSize = FLOAT_TYPE_SIZE * inputImageWidth *
                inputImageHeight * PIXEL_SIZE

        val outputShape = tfLite!!.getOutputTensor(0).shape()
        outputImageWidth = outputShape[1]
        outputImageHeight = outputShape[2]
        modelOutputSize = FLOAT_TYPE_SIZE * outputImageWidth *
                outputImageHeight * PIXEL_SIZE

        Log.d("DEBUG_POINT", "DEBUG_POINT")
    }

    fun imageToBitmap() {
//        val d : Drawable
//        d = context.resources.getDrawable(R.drawable.detected_lanes)
//        val bitmap = Bitmap.createBitmap(
//            d.intrinsicHeight,
//            d.intrinsicWidth,
//            Bitmap.Config.ARGB_8888
//        )
        //val path : String = getExternalStorageDirectory().getAbsolutePath() + "/detected_lanes.png"
        var imgMat = Imgcodecs.imread("/storage/emulated/0/Download/detected_lanes.jpg")
        Imgproc.cvtColor(imgMat, imgMat, Imgproc.COLOR_BGR2RGB)
//        val tempBitmap : Bitmap
//        org.opencv.android.Utils.matToBitmap(imgMat,tempBitmap)
        //val bitmap = BitmapFactory.decodeResource(context.resources, R.drawable.detected_lanes)
        Log.d("Image","Mat: "+imgMat.toString())
//        Log.i("bitmap",$bitmap)
        classifyAsync(imgMat)

    }

    fun classify(mat: Mat): String {
        check(isInitialized) { "TF Lite Interpreter is not initialized yet." }

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
        val imgSize = Size(inputImageWidth.toDouble(), inputImageHeight.toDouble())
        var resizedImage = Imgproc.resize(mat, mat, imgSize)

        Log.d("ResizedImage","resized image: "+resizedImage.toString())
        //resizedImage = resizedImage(DataType.FLOAT32)X

//        val bytes = FloatArray(mat.rows() * mat.cols() * mat.channels())
//
//        mat.convertTo(mat,CV_32FC3)
//        mat.get(0,0,bytes)

        //var byteBuffer = ByteBuffer.wrap(bytes).asFloatBuffer()
        //var byteBuffer = FloatBuffer.wrap(bytes)


        var bitmap = Bitmap.createBitmap(inputImageWidth, inputImageHeight, Bitmap.Config.ARGB_8888)
        matToBitmap(mat,bitmap)

        var byteBuffer = convertBitmapToByteBuffer(bitmap)
        Log.d("bytebuffer","bytebuffer: "+byteBuffer.toString())
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
        byteBuffer!!.rewind()
        tfLite!!.run(byteBuffer, output)
        Log.d("Output after interpreter","output: "+output.contentDeepToString())
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
        return ArrayList()
    }

    override fun close() {
        executorService.execute {
            tfLite?.close()
            Log.d(TAG, "Closed TFLite interpreter.")
        }
    }

    override fun setNumOfThreads(numThread: Int) {
        if (tfLite != null) tfLite!!.setNumThreads(numThread)
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

    private fun convertBitmapToByteBuffer(bitmap: Bitmap?): ByteBuffer? {
        val byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * 288 * 800 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(288*800)
        bitmap!!.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val pixel = 0
        //imgData!!.rewind()
        for (i in 0 until INPUT_HEIGHT) {
            for (j in 0 until INPUT_WIDTH) {
                val pixelValue = intValues[i * INPUT_WIDTH + j]
                if (isModelQuantized) {
                    // Quantized model
                    byteBuffer.put((((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point) as Byte)
                    byteBuffer.put((((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point) as Byte)
                    byteBuffer.put((((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point) as Byte)
                } else { // Float model
                    byteBuffer.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    byteBuffer.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    byteBuffer.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                }
            }
        }
        return byteBuffer
    }


    /*private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(modelInputSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputImageWidth * inputImageHeight)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixelValue in pixels) {
            val r = (pixelValue shr 16 and 0xFF)
            val g = (pixelValue shr 8 and 0xFF)
            val b = (pixelValue and 0xFF)

            // Convert RGB to grayscale and normalize pixel value to [0..1].
            val normalizedPixelValue = (r + g + b) / 3.0f / 255.0f
            byteBuffer.putFloat(normalizedPixelValue)
        }

        return byteBuffer
    }*/

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


    @RequiresApi(Build.VERSION_CODES.N)
    private fun softmax(input: Double, neuronValues: DoubleArray): Double {
        val total = Arrays.stream(neuronValues).map { v: Double ->
            Math.exp(
                v
            )
        }.sum()
        return Math.exp(input) / total
    }


    fun processOutput(imgData: Array<Array<Array<FloatArray>>>)
    {

        var processed_output = mutableListOf<MutableList<MutableList<Float>>>()

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

//        Log.d("Prob","prob: "+prob.toString())

        val griding_num = mk.arange<Int>(1,101,1)                           // create array 1-100

//        Log.d("Griding Number","griding_num: "+griding_num.toString())

        val idx = griding_num.reshape(griding_num.shape[0],1,1)
//        Log.d("IDX","idx: "+idx.toString())

        //var loc_prob_idx = prob.asD3Array() * idx.asType<Float>()


        //need to fix append (not sure why it is not working)
//        var loc_prob_idx = mk.zeros<Float>(prob.shape[0],prob.shape[1],prob.shape[2])
//
//        for(i in 0..(prob.shape[0]-1)) {
//            for (j in 0..(prob.shape[1] - 1)) {
//                for (k in 0..(prob.shape[2] - 1)) {
//                    Log.d("PROB*IDX","prob*idx: "+(prob[i][j][k] * idx[i][0][0]).toString())
//                    loc_prob_idx[i].append((prob[i][j][k] * idx[i][0][0]))
//                }
//            }
//        }


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


//        Log.d("Loc","loc: "+loc.toString())

        //processed_output_ndarry = mk.math.argMax(processed_output_ndarry.asD3Array(), axis = 0)
        var argMax_process_output = mk.math.argMaxD3(processed_output_ndarry.asD3Array(), axis = 0)

        Log.d("argMax","argMax: "+argMax_process_output.toString())

        //loc[argMax_process_output == 100] = 0                        //convert to kotlin

        //need to fix append (not sure why it is not working)
//        val argMaxSize = argMax_process_output.shape[0]
//        val argMaxHeight = argMax_process_output.shape[1]
//        for (i in 0..(argMaxSize-1))
//            for(j in 0..(argMaxHeight-1))
//                if(argMax_process_output[i][j].toFloat() != 100F)
//                {
//                    loc[i].append(argMax_process_output[i][j].toFloat())
//                }
//                else
//                {
//                    loc[i].append(0F)
//                }

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
                        Log.d("Lane point left","lane_point_left: "+lane_point_left.toString())


//                        Log.d("tusimple_row_anchor1","tusimple_row_anchor1: "+(tusimple_row_anchor[55-point_num]/288).toString())
//                        Log.d("tusimple_row_anchor2","tusimple_row_anchor2: "+((tusimple_row_anchor[55-point_num]/288)*720).toString())
                        var lane_point_right = (720 * (tusimple_row_anchor[55-point_num].toFloat() / 288)).toInt() - 1
                        Log.d("Lane point right","lane_point_right: "+lane_point_right.toString())


                        var lane_point = arrayListOf(lane_point_left,lane_point_right)
                        Log.d("Lane point","lane_point: "+lane_point.toString())
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
            assetManager: AssetManager,
            modelFilename: String?,
            labelFilename: String,
            isQuantized: Boolean,
            inputWidth: Int,
            inputHeight: Int
        ): LaneClassifier {
            val d = LaneClassifier(AppCompatActivity())
            Log.d("FileName: ", "$modelFilename")
            val actualFilename = labelFilename.split("file:///android_asset/")[1]
            val labelsInput = assetManager.open(actualFilename)
            val br = BufferedReader(InputStreamReader(labelsInput))
            br.forEachLine {
                Log.d("Label: ", it)
                d.labels.add(it)
            }
            br.close()
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
                d.tfliteModel = Utils.loadModelFile(assetManager, modelFilename)
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
            d.imgData = ByteBuffer.allocateDirect(1 * 80 * 160 * 3 * numBytesPerChannel)
            d.imgData!!.order(ByteOrder.nativeOrder())
            d.intValues = IntArray(80 * 160)
            /*d.output_box =
                (((inputSize / 32.0f).pow(2.0f) + (inputSize / 16.0f).pow(2.0f) + (inputSize / 8.0f).pow(2.0f)) * 3).toInt()*/
            Log.d("ImgData", "${d.INPUT_WIDTH} x ${d.INPUT_HEIGHT}")
            if (d.isModelQuantized) {
                val inpten: Tensor = d.tfLite!!.getInputTensor(0)
                d.inp_scale = inpten.quantizationParams().scale
                d.inp_zero_point = inpten.quantizationParams().zeroPoint
                val oupten: Tensor = d.tfLite!!.getOutputTensor(0)
                d.oup_scale = oupten.quantizationParams().scale
                d.oup_zero_point = oupten.quantizationParams().zeroPoint
            }
            val shape: IntArray = d.tfLite!!.getOutputTensor(0).shape()
            val numClass = shape[shape.size - 1] - 5
            Log.d("NumClass", "$numClass")
            d.numClass = numClass
            d.outData = ByteBuffer.allocateDirect(80 * 160 * numBytesPerChannel)
            d.outData!!.order(ByteOrder.nativeOrder())
            return d
        }

    }
}