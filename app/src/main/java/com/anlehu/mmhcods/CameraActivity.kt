package com.anlehu.mmhcods

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.Image
import android.media.ImageReader
import android.os.*
import android.util.Log
import android.util.Size
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import com.anlehu.mmhcods.utils.ImageUtils

abstract class CameraActivity: AppCompatActivity(),
    ImageReader.OnImageAvailableListener {

    private var handlerThread: HandlerThread? = null
    private var handler: Handler? = null
    private val yuvBytes = arrayOfNulls<ByteArray>(3)
    private var yRowStride: Int = 0
    private var isProcessingFrame: Boolean = false
    private var screenPreviewWidth = 0
    private var screenPreviewHeight = 0

    private lateinit var postInferenceCallback: Runnable

    var rgbBytes: IntArray? = null
    lateinit var imageConverter: Runnable

    override fun onCreate(savedInstanceState: Bundle?){
        super.onCreate(null)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.camera_detection_activity)

        if(permissionGranted()){
            showCameraFragment()
        }else{
            requestCameraPermissions()
        }
    }

    override fun onImageAvailable(imageReader: ImageReader) {

        // Only run if the preview size has been initialized
        if(screenPreviewWidth == 0 || screenPreviewHeight == 0){

            return
        }
        if(rgbBytes == null){
            // initialize rgbBytes if still null
            rgbBytes = IntArray(screenPreviewWidth * screenPreviewHeight)
        }
        try{
            // acquire the latest image from the image reader, return if no image
            val image = imageReader.acquireLatestImage() ?: return
            
            // if a frame is still being processed, do not process new frame
            if(isProcessingFrame){
                image.close()
                return
            }
            // now lock the processing
            isProcessingFrame = true
            Trace.beginSection("IMAGE_AVAILABLE")
            val planes: Array<Image.Plane> = image.planes
            fillBytes(planes, yuvBytes as Array<ByteArray>)
            // Get the YUV image strides
            yRowStride = planes[0].rowStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride

            imageConverter = Runnable{
                ImageUtils.convertYUVtoARGB(
                    yuvBytes[0]!!,
                    yuvBytes[1]!!,
                    yuvBytes[2]!!,
                    screenPreviewWidth,
                    screenPreviewHeight,
                    yRowStride,
                    uvRowStride,
                    uvPixelStride,
                    rgbBytes as IntArray
                )
            }
            // close image and set isProcessing frame to false
            postInferenceCallback = Runnable{
                image.close()
                isProcessingFrame = false
            }

            processImage()
        }catch(e: Exception){
            Log.e("CAM_ACT_ERROR: ", e.toString())
            Trace.endSection()
            return
        }
        Trace.endSection()
    }

    @Synchronized
    fun runInBackground(r: Runnable){
        if(handler != null)
            handler!!.post(r)
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

    fun readyForNextImage(){
        postInferenceCallback.run()
    }

    private fun fillBytes(planes: Array<Image.Plane>, yuvBytes: Array<ByteArray>) {
        for(i in planes.indices){
            val byteBuffer = planes[i].buffer
            if(yuvBytes[i] == null){
                Log.d("CAM_ACT_INFO: ", "fillBuffer: $i at ${byteBuffer.capacity()}")
                yuvBytes[i] = ByteArray(byteBuffer.capacity())
            }
            byteBuffer.get(yuvBytes[i]!!)
        }
    }

    fun getRGBBytes(): IntArray{
        imageConverter.run()
        return rgbBytes!!
    }

    private fun showCameraFragment() {
        val cameraId = chooseCamera()
        val cameraFragment: CameraFragment = CameraFragment.newInstance(
            { size, cameraRotation ->
                screenPreviewHeight = size.height
                screenPreviewWidth = size.width
                this.onPreviewSizeChosen(size, cameraRotation)
            },
            this,
            getLayoutId(),
            getDesiredPreviewFrameSize())

        cameraFragment.setCamera(cameraId as String)
        val fragment: Fragment = cameraFragment
        supportFragmentManager.beginTransaction().replace(R.id.container, fragment).commit()
    }

    private fun chooseCamera(): String? {
        val manager: CameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try{
            for(cameraId in manager.cameraIdList){
                val characteristics = manager.getCameraCharacteristics(cameraId)
                val facing: Int = characteristics.get(CameraCharacteristics.LENS_FACING)!!.toInt()
                if(facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT){
                    continue
                }
                return cameraId
            }
        }catch(e: Exception){throw Exception(e)}

        return null
    }

    private fun requestCameraPermissions(){
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
            if(shouldShowRequestPermissionRationale(PERMISSION_CAMERA)){
                Toast.makeText(this, "Requesting Camera Permissions", Toast.LENGTH_SHORT).show()
            }
            val permissions = arrayOf(PERMISSION_CAMERA)
            requestPermissions(permissions, PERMISSIONS_REQUEST)
        }
        return
    }

    private fun permissionGranted(): Boolean {
        return if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
            checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED
        }else{
            true
        }
    }

    abstract fun getLayoutId(): Int
    abstract fun getDesiredPreviewFrameSize(): Size
    abstract fun onPreviewSizeChosen(size: Size, rotation: Int)
    abstract fun processImage()

    companion object{

        var PERMISSION_CAMERA = Manifest.permission.CAMERA
        const val PERMISSIONS_REQUEST = 1

    }


}
