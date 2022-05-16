package com.anlehu.mmhcods

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import com.anlehu.mmhcods.utils.Detector
import com.anlehu.mmhcods.utils.ModelUtils
import org.checkerframework.checker.nullness.qual.NonNull
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.util.*

class LaneClassifier: Detector {

    // Config values.
    // Pre-allocated buffers.
    private var lane = object {
        var recentFit:MutableList<Array<FloatArray>> = ArrayList()
        var avgFit: List<FloatArray> = ArrayList()
    }

    private val labels = Vector<String>()
    private var intValues: IntArray = IntArray(1)

    private var imgData: ByteBuffer? = null
    private var outData: ByteBuffer? = null

    private var tfLite: Interpreter? = null
    private var inp_scale = 0f
    private var inp_zero_point = 0
    private var oup_scale = 0f
    private var oup_zero_point = 0
    private var numClass = 0

    override fun detectImage(bitmap: Bitmap): ArrayList<Detector.Detection> {
        return ArrayList()
    }

    override fun close() {
        tfLite!!.close()
        tfLite = null
        if (gpuDelegate != null) {
            gpuDelegate!!.close()
            gpuDelegate = null
        }
        if (nnapiDelegate != null) {
            nnapiDelegate!!.close()
            nnapiDelegate = null
        }
        tfliteModel = null
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
            tfLite = Interpreter(tfliteModel as @NonNull ByteBuffer, tfliteOptions)
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

    /*
    //non maximum suppression
    @RequiresApi(Build.VERSION_CODES.N)
    open fun nms(list: ArrayList<Detector.Detection>): ArrayList<Detector.Detection> {
        val nmsList = ArrayList<Detector.Detection>()
        for (k in labels.indices) {
            //1.find max confidence per class
            val pq: PriorityQueue<Detector.Detection> = PriorityQueue<Detector.Detection>(DetectionComparator)
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
    */

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

    /**
     * Writes Image data into a `ByteBuffer`.
     */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap){
//        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
//        byteBuffer.order(ByteOrder.nativeOrder());
//        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val pixel = 0
        imgData!!.rewind()
        for (i in 0 until INPUT_HEIGHT) {
            for (j in 0 until INPUT_WIDTH) {
                val pixelValue = intValues[i * INPUT_WIDTH + j]
                if (isModelQuantized) {
                    // Quantized model
                    imgData!!.put((((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point) as Byte)
                    imgData!!.put((((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point) as Byte)
                    imgData!!.put((((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point) as Byte)
                } else { // Float model
                    imgData!!.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData!!.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData!!.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                }
            }
        }//return imgData
    }
    //fun detectLane(bitmap: Bitmap): Array<FloatArray> {
    @RequiresApi(Build.VERSION_CODES.N)
    fun detectLane(bitmap: Bitmap): FloatArray {

        convertBitmapToByteBuffer(bitmap)

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

    companion object{

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
            val d = LaneClassifier()
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
                d.tfliteModel = ModelUtils.loadModelFile(assetManager, modelFilename)
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