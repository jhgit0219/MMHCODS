/*  THIS FILE HAS BEEN CONVERTED TO KOTLIN AND MODIFIED
    TO SUIT THE APPLICATION REQUIREMENTS UNDER APACHE LICENSE 2.0.
    ORIGINAL FILE COPYRIGHT IS SHOWN BELOW:
==========================================================================*/

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.anlehu.mmhcods

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import com.anlehu.mmhcods.utils.DetectionComparator
import com.anlehu.mmhcods.utils.Detector
import com.anlehu.mmhcods.utils.Detector.Detection
import com.anlehu.mmhcods.utils.ModelUtils
import org.checkerframework.checker.nullness.qual.NonNull
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.util.*
import kotlin.math.pow

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * - https://github.com/tensorflow/models/tree/master/research/object_detection
 * where you can find the training code.
 * <p>
 * To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
 * - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 */
open class YoloV5Classifier: Detector {

    // Config values.
    // Pre-allocated buffers.
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

    fun getInputSize(): Int {
        return INPUT_SIZE
    }

    fun enableStatLogging(logStats: Boolean) {}

    fun getStatString(): String? {
        return ""
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
        if (tfLite != null) tfliteOptions.numThreads = numThread
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

    //config yolo
    private var INPUT_SIZE: Int = -1

    //    private int[] OUTPUT_WIDTH;
    //    private int[][] MASKS;
    //    private int[] ANCHORS;
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

    //non maximum suppression
    @RequiresApi(Build.VERSION_CODES.N)
    open fun nms(list: ArrayList<Detection>): ArrayList<Detection> {
        val nmsList = ArrayList<Detection>()
        for (k in labels.indices) {
            //1.find max confidence per class
            val pq: PriorityQueue<Detection> = PriorityQueue<Detection>(DetectionComparator)
            for (i in list.indices) {
                if (list[i].detectedClass == k) {
                    pq.add(list[i])
                }
            }

            //2.do non maximum suppression
            while (pq.size > 0) {
                //insert detection with max confidence
                val a: Array<Detection?> = arrayOfNulls(pq.size)
                val detections: Array<Detection> = pq.toArray(a)
                val max: Detection = detections[0]
                nmsList.add(max)
                pq.clear()
                for (j in 1 until detections.size) {
                    val detection: Detection = detections[j]
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

    /**
     * Writes Image data into a `ByteBuffer`.
     */
    private fun convertBitmapToByteBuffer(bitmap: Bitmap){
//        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
//        byteBuffer.order(ByteOrder.nativeOrder());
//        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        imgData!!.rewind()
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                /**
                 *  The "stride" value is equal to the height value because this is a flat array
                 *  e.g. Each height (i) contains a value from index 0 to width. Traversing to the next i,
                 *  just have to multiply i by the width ( next i, j=0)
                 */

                val pixelValue = intValues[i * INPUT_SIZE + j]
                if (isModelQuantized) {
                    // Quantized model
                    imgData!!.put(
                        (((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point).toInt().toByte()
                    )
                    imgData!!.put((((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point).toInt().toByte())
                    imgData!!.put((((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD / inp_scale + inp_zero_point).toInt().toByte())
                } else { // Float model
                    imgData!!.putFloat(((pixelValue shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData!!.putFloat(((pixelValue shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    imgData!!.putFloat(((pixelValue and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                }
            }
        }//return imgData
    }

    @RequiresApi(Build.VERSION_CODES.N)
    override fun detectImage(bitmap: Bitmap): ArrayList<Detection> {

        convertBitmapToByteBuffer(bitmap)

        val outputMap: MutableMap<Int?, Any?> = HashMap()

        outData!!.rewind()
        outputMap[0] = outData
        val inputArray: Array<out Any> = arrayOf(imgData as Any)
        tfLite!!.runForMultipleInputsOutputs(inputArray, outputMap)
        val byteBuffer = outputMap[0] as ByteBuffer
        byteBuffer.rewind()
        val detections: ArrayList<Detection> = ArrayList()

        val out = Array(1) {
            Array(output_box) {
                FloatArray(
                    numClass + 5
                )
            }
        }
        Log.d("YoloV5Classifier", "out[0] detect start")
        for (i in 0 until output_box) {
            for (j in 0 until numClass + 5) {
                if (isModelQuantized) {
                    out[0][i][j] = oup_scale * ((byteBuffer.get().toInt() and 0xFF) - oup_zero_point)
                } else {
                    out[0][i][j] = byteBuffer.float
                }
            }
            // Denormalize xywh
            for (j in 0 until 4) {
                out[0][i][j] *= INPUT_SIZE.toFloat()
            }
        }
        for (i in 0 until output_box) {
            val offset = 0
            val confidence = out[0][i][4]
            var detectedClass = -1
            var maxClass = 0f
            val classes = FloatArray(labels.size)
            for (c in labels.indices) {
                classes[c] = out[0][i][5 + c]
            }
            for (c in labels.indices) {
                if (classes[c] > maxClass) {
                    detectedClass = c
                    maxClass = classes[c]
                }
            }
            val confidenceInClass = maxClass * confidence
            if (confidenceInClass > getObjThresh()) {
                val xPos = out[0][i][0]
                val yPos = out[0][i][1]
                val w = out[0][i][2]
                val h = out[0][i][3]
                val rect = RectF(
                    Math.max(0f, xPos - w / 2),
                    Math.max(0f, yPos - h / 2),
                    Math.min((bitmap.width - 1).toFloat(), xPos + w / 2),
                    Math.min((bitmap.height - 1).toFloat(), yPos + h / 2)
                )
                detections.add(
                    Detection(
                        "" + offset, labels[detectedClass],
                        confidenceInClass, rect, detectedClass
                    )
                )
            }
        }
        //        final ArrayList<Detection> recognitions = detections;
        return nms(detections)
    }

    open fun checkInvalidateBox(
        x: Float,
        y: Float,
        width: Float,
        height: Float,
        oriW: Float,
        oriH: Float,
        intputSize: Int
    ): Boolean {
        // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        val halfHeight = height / 2.0f
        val halfWidth = width / 2.0f
        val pred_coor = floatArrayOf(x - halfWidth, y - halfHeight, x + halfWidth, y + halfHeight)

        // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        val resize_ratioW = 1.0f * intputSize / oriW
        val resize_ratioH = 1.0f * intputSize / oriH
        val resize_ratio = if (resize_ratioW > resize_ratioH) resize_ratioH else resize_ratioW //min
        val dw = (intputSize - resize_ratio * oriW) / 2
        val dh = (intputSize - resize_ratio * oriH) / 2
        pred_coor[0] = 1.0f * (pred_coor[0] - dw) / resize_ratio
        pred_coor[2] = 1.0f * (pred_coor[2] - dw) / resize_ratio
        pred_coor[1] = 1.0f * (pred_coor[1] - dh) / resize_ratio
        pred_coor[3] = 1.0f * (pred_coor[3] - dh) / resize_ratio

        // (3) clip some boxes those are out of range
        pred_coor[0] = if (pred_coor[0] > 0) pred_coor[0] else 0f
        pred_coor[1] = if (pred_coor[1] > 0) pred_coor[1] else 0f
        pred_coor[2] = if (pred_coor[2] < oriW - 1) pred_coor[2] else oriW - 1f
        pred_coor[3] = if (pred_coor[3] < oriH - 1) pred_coor[3] else oriH - 1f
        if (pred_coor[0] > pred_coor[2] || pred_coor[1] > pred_coor[3]) {
            pred_coor[0] = 0f
            pred_coor[1] = 0f
            pred_coor[2] = 0f
            pred_coor[3] = 0f
        }

        // (4) discard some invalid boxes
        val temp1 = pred_coor[2] - pred_coor[0]
        val temp2 = pred_coor[3] - pred_coor[1]
        val temp = temp1 * temp2
        if (temp < 0) {
            Log.e("checkInvalidateBox", "temp < 0")
            return false
        }
        if (Math.sqrt(temp.toDouble()) > Float.MAX_VALUE) {
            Log.e("checkInvalidateBox", "temp max")
            return false
        }
        return true
    }
    companion object{
        var mountedPath = ""
        // Number of threads in the java app
        val NUM_THREADS = 2
        val isNNAPI = false
        val isGPU = true
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
            labelFilename: String,
            isQuantized: Boolean,
            inputSize: Int
        ): YoloV5Classifier {
            val d = YoloV5Classifier()
            Log.d("FileName: ", "$modelFilename")
            //replace with OBB
            // val labelsInput = assetManager.open(actualFilename)
            Log.d("CREATING MODEL", "$mountedPath/$labelFilename")
            val labelsInput = FileInputStream(File("$mountedPath/$labelFilename"))
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
                    /*val gpu_options: GpuDelegate.Options = GpuDelegate.Options()
                    gpu_options.setPrecisionLossAllowed(true)
                    gpu_options.setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                    d.gpuDelegate = GpuDelegate(gpu_options)
                    options.addDelegate(d.gpuDelegate)*/
                    options.apply{
                        this.addDelegate(GpuDelegate(CompatibilityList().bestOptionsForThisDevice))
                    }
                }
                d.tfliteModel = ModelUtils.loadModelFile(mountedPath, modelFilename)
                d.tfLite = Interpreter(d.tfliteModel!!, options)
            } catch (e: Exception) {
                throw RuntimeException(e)
            }
            d.isModelQuantized = isQuantized
            Log.d("YoloV5Classifier", "Quantized: $isQuantized")
            // Pre-allocate buffers.
            val numBytesPerChannel: Int = if (isQuantized) {
                1 // Quantized
            } else {
                4 // Floating point
            }
            d.INPUT_SIZE = inputSize
            d.imgData = ByteBuffer.allocateDirect(1 * d.INPUT_SIZE * d.INPUT_SIZE * 3 * numBytesPerChannel)
            d.imgData!!.order(ByteOrder.nativeOrder())
            d.intValues = IntArray(d.INPUT_SIZE * d.INPUT_SIZE)
            d.output_box =
                (((inputSize / 32.0f).pow(2.0f) + (inputSize / 16.0f).pow(2.0f) + (inputSize / 8.0f).pow(2.0f)) * 3).toInt()
            Log.d("ImgData", "${d.INPUT_SIZE}")
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
            d.outData = ByteBuffer.allocateDirect(d.output_box * (numClass + 5) * numBytesPerChannel)
            d.outData!!.order(ByteOrder.nativeOrder())
            return d
        }
    }
}