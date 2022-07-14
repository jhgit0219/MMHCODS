package com.anlehu.mmhcods

import android.content.Context
import android.content.res.Configuration
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.*
import android.hardware.camera2.CameraCaptureSession.CaptureCallback
import android.hardware.camera2.params.StreamConfigurationMap
import android.media.ImageReader
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.util.SparseIntArray
import android.view.*
import androidx.fragment.app.Fragment
import com.anlehu.mmhcods.views.AutoAdjustView
import java.util.*
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit

class CameraFragment() : Fragment() {

    /********************************************************************************************************
     * Variable Initializaitons
     ********************************************************************************************************/
    private lateinit var adjustView: AutoAdjustView
    private lateinit var previewSize: Size
    private lateinit var imageReader: ImageReader
    private lateinit var previewRequestBuilder: CaptureRequest.Builder
    private lateinit var captureRequest: CaptureRequest
    private lateinit var inputSize: Size

    private lateinit var cameraConnectionCallback: ConnectionCallback
    private lateinit var imageListener: ImageReader.OnImageAvailableListener
    private var orientation_array = SparseIntArray()

    private var cameraDevice: CameraDevice? = null
    private var cameraId: String = ""
    private var sensorOrientation: Int = 0
    private var captureSession: CameraCaptureSession? = null
    private var previewReader: ImageReader? = null
    private var backgroundThread: HandlerThread? = null
    private var backgroundHandler: Handler? = null
    private var layout: Int = 0

    private var MINIMUM_PREVIEW_SIZE = 640

    /** Semaphore to lock camera; won't allow app to exit before camera closes**/
    var cameraLock: Semaphore = Semaphore(1)

    /********************************************************************************************************
     * Function that tracks progress of Capture Request. Sets session, request, and result.
     ********************************************************************************************************/
    private var captureCallback: CaptureCallback =
        object: CaptureCallback() {
            override fun onCaptureProgressed(
                session: CameraCaptureSession,
                request: CaptureRequest,
                partialResult: CaptureResult){}
            override fun onCaptureCompleted(
                session: CameraCaptureSession,
                request: CaptureRequest,
                result: TotalCaptureResult) {}
    }
    /********************************************************************************************************
     * Function Constructor
     ********************************************************************************************************/
    constructor(
        callback: ConnectionCallback,
        imageAvailableListener: ImageReader.OnImageAvailableListener,
        layout: Int,
        inputSize: Size) : this() {
        this.cameraConnectionCallback = callback
        this.imageListener = imageAvailableListener
        this.layout = layout
        this.inputSize = inputSize
    }
    /********************************************************************************************************
     * Get Capture Callback Object
     * @return current captureCallback instance
     ********************************************************************************************************/
    fun getThisCaptureCallback(): CaptureCallback{
        return captureCallback
    }
    /********************************************************************************************************
     * Function that performs certain actions depending on status of current capture camera session
     ********************************************************************************************************/
    private var stateCallback =
        object: CameraDevice.StateCallback() {
            override fun onOpened(camDevice: CameraDevice){
                cameraLock.release()
                cameraDevice = camDevice
                createCameraSession()
            }

            override fun onDisconnected(camDevice: CameraDevice) {
                cameraLock.release()
                camDevice.close()
                cameraDevice = null
            }

            override fun onError(camDevice: CameraDevice, error: Int) {
                cameraLock.release()
                camDevice.close()
                cameraDevice = null
                activity?.finish()
            }
        }

    /********************************************************************************************************
     * Function that performs certain actions depending on the status of
     * the associated surface texture in the current texture view
     ********************************************************************************************************/
    var surfaceTextureListener: TextureView.SurfaceTextureListener =
        object: TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                Log.d("SurfaceTexture", "Opening Camera")
                openCamera(p1, p2)
            }

            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
                configTransform(p1, p2)
            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return true
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
            }
        }

    /********************************************************************************************************
     * Function that determines optimal size for camera output
     ********************************************************************************************************/
    private fun chooseOptimalSize(choices: Array<Size>, width: Int, height: Int): Size{
        Log.d("DEBUG", "SUCCESS")
        val minSize = Math.max(Math.min(width.toFloat(), height.toFloat()), MINIMUM_PREVIEW_SIZE.toFloat())
        var desiredSize = Size(width, height)
        var sizes: MutableList<Size> = ArrayList()

        var exactSizeFound = false
        for(option in choices){
            Log.d("RESO", "${option.width} x ${option.height}")
            if(option == desiredSize)
                exactSizeFound = true
            if(option.height >= minSize && option.width >= minSize)
                sizes.add(option)
        }
        if(exactSizeFound){
            return desiredSize
        }
        return if(sizes.size > 0) {
            Collections.min(sizes, CompareSizesByArea())
        } else{
            choices[0]
        }
    }
    /********************************************************************************************************
     * Function that returns on create view
     * @return attributes for on create view
     ********************************************************************************************************/
    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        return inflater.inflate(layout, container, false)
    }
    /********************************************************************************************************
     * Function that sets AutoAdjustView
     ********************************************************************************************************/
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        adjustView = view.findViewById(R.id.adjust_view)
    }
    /********************************************************************************************************
     * Prints status of adjustView once process is resumed
     ********************************************************************************************************/
    override fun onResume() {
        super.onResume()
        startBackgroundThread()

        if(adjustView.isAvailable){
            Log.d("RESUMING", "OPENING CAMERA")
            openCamera(adjustView.width, adjustView.height)
        }else{
            Log.d("RESUMING", "ADDING TEXTURE LISTENER")
            adjustView.surfaceTextureListener = surfaceTextureListener
        }
    }
    /********************************************************************************************************
     * Function that handles actions that happen when process is paused
     ********************************************************************************************************/
    override fun onPause() {
        Log.d("CLOSING", "SUCCESS")
        closeCamera()
        stopBackgroundThread()
        super.onPause()
    }
    /********************************************************************************************************
     * Set the camera ID
     ********************************************************************************************************/
    fun setCamera(cameraId: String){this.cameraId = cameraId}
    /********************************************************************************************************
     * Function that sets camera output
     ********************************************************************************************************/
    private fun setCameraOutputs(){
        Log.d("DEBUG", "Setting up Camera Outputs")
        var activity = activity
        var camManager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try{
            var cameraCharacteristics = camManager.getCameraCharacteristics(cameraId)
            var map: StreamConfigurationMap? = cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)

            sensorOrientation = cameraCharacteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)!!.toInt()
            previewSize = chooseOptimalSize(map!!.getOutputSizes(SurfaceTexture::class.java), inputSize.width, inputSize.height)
            //previewSize = Size(1920, 1080)
            val orientation = resources.configuration.orientation
            if(orientation == Configuration.ORIENTATION_LANDSCAPE){
                adjustView.setAspectRatio(previewSize.width, previewSize.height, true)
            }else{
                adjustView.setAspectRatio(previewSize.height, previewSize.width, true)
            }
        }catch(e: Exception){
            throw IllegalStateException(e)
        }
        cameraConnectionCallback.onPreviewSizeChosen(previewSize, sensorOrientation)
    }
    /********************************************************************************************************
     * Function that opens the camera for inputs
     ********************************************************************************************************/
    fun openCamera(width: Int, height: Int){
        Log.d("Opening Camera", "SUCCESS")
        setCameraOutputs()
        configTransform(width, height)
        val activity = activity
        val manager: CameraManager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try{
            if(!cameraLock.tryAcquire(3000, TimeUnit.MILLISECONDS)){
                throw RuntimeException("Timeout while waiting to lock camera")
            }
            manager.openCamera(cameraId, stateCallback, backgroundHandler)
        }catch(e: Exception){
            throw Exception(e)
        }
    }
    /********************************************************************************************************
     * Function that closes camera and camera session once application is stopped
     ********************************************************************************************************/
    private fun closeCamera(){
        try{
            cameraLock.acquire()
            if(captureSession != null){
                captureSession!!.close()
                captureSession = null
                Log.d("CLOSING: CAP_SESS", "Successful")
            }
            if(cameraDevice != null){
                cameraDevice!!.close()
                cameraDevice = null
                Log.d("CLOSING: CAM_DEV", "Successful")
            }
            if(previewReader != null){
                previewReader!!.close()
                previewReader = null
                Log.d("CLOSING: PREV_READ", "Successful")
            }
        }catch (e: Exception){
            Log.d("StopThread","An error occurred while closing camera")
        }finally{
            cameraLock.release()
            Log.d("CAMLOCK", "RELEASED Successful")
        }
    }
    /********************************************************************************************************
     * Start background thread for ImageListener
     ********************************************************************************************************/
    private fun startBackgroundThread(){
        backgroundThread = HandlerThread("ImageListener")
        backgroundThread!!.start()
        backgroundHandler = Handler(backgroundThread!!.looper)
        Log.d("START_THREAD", "STARTED")
    }
    /********************************************************************************************************
     * Stops current running background thread
     ********************************************************************************************************/
    private fun stopBackgroundThread(){
        backgroundThread!!.quitSafely()
        try{
            backgroundThread!!.join()
            backgroundThread = null
            backgroundHandler = null
        }catch(e: Exception){
            Log.d("StopThread", e.toString())
        }
    }

    /********************************************************************************************************
     * Function that initializes attributes for the creation of new Camera Session
     ********************************************************************************************************/
    private fun createCameraSession(){
        try{
            val texture = adjustView.surfaceTexture
            assert(texture != null)

            // Configure size of buffer to be equal to camera preview size
            texture!!.setDefaultBufferSize(previewSize.width, previewSize.height)
            var surface = Surface(texture)

            previewRequestBuilder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
            previewRequestBuilder.addTarget(surface)

            previewReader = ImageReader.newInstance(previewSize.width, previewSize.height, ImageFormat.YUV_420_888, 2)
            previewReader!!.setOnImageAvailableListener(imageListener, backgroundHandler)
            previewRequestBuilder.addTarget(previewReader!!.surface)


            cameraDevice!!.createCaptureSession(
                listOf(surface, previewReader!!.surface),
                object: CameraCaptureSession.StateCallback() {
                    override fun onConfigured(p0: CameraCaptureSession) {
                        if(cameraDevice == null){
                            return
                        }

                        // If session is ready

                        captureSession = p0
                        try{
                            previewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                            // enable flash automatically
                            previewRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH)

                            // Display camera preview with configurations above
                            captureRequest = previewRequestBuilder.build()
                            captureSession!!.setRepeatingRequest(captureRequest, getThisCaptureCallback(), backgroundHandler)
                        }catch(e: Exception){
                            Log.e("OnConfigured", "Error occurred while creating capture session")
                        }
                    }

                    override fun onConfigureFailed(p0: CameraCaptureSession) {
                    }
                },
                null)
        }catch (e: Exception){
            Log.e("OnConfigured", "Error occurred while creating capture session")
        }
    }
    /********************************************************************************************************
     * Configures SurfaceTextureListener used to listen to surface texture events in camera session
     ********************************************************************************************************/
    fun configTransform(width: Int, height: Int){
        val activity = activity
        if(adjustView == null || previewSize == null || activity == null){
            return
        }
        val rotation: Int = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            activity.display!!.rotation
        } else {
            @Suppress("DEPRECATION")
            activity.windowManager.defaultDisplay.rotation
        }
        val matrix = Matrix()
        val viewRect = RectF(0f, 0f, width.toFloat(), height.toFloat())
        val bufferRect = RectF(0f, 0f, previewSize.height.toFloat(), previewSize.width.toFloat())
        val centerX = viewRect.centerX()
        val centerY = viewRect.centerY()
        if(rotation == Surface.ROTATION_90 || rotation == Surface.ROTATION_270){
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY())
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL)
            val scale = Math.max((height.toFloat() / previewSize.height.toFloat()), (width.toFloat() / previewSize.width.toFloat()))
            matrix.postScale(scale, scale, centerX, centerY)
            matrix.postRotate((90 * (rotation -2)).toFloat(), centerX, centerY)
        }else if(rotation == Surface.ROTATION_180){
            matrix.postRotate(180f, centerX, centerY)
        }
        adjustView.setTransform(matrix)
    }
    /********************************************************************************************************
     * Companion Object
     * Sets new instance of CameraFragment
     ********************************************************************************************************/
    companion object{
        fun newInstance(
            callback: ConnectionCallback,
            imageAvailableListener: ImageReader.OnImageAvailableListener,
            layout: Int,
            inputSize: Size): CameraFragment{

            return CameraFragment(callback, imageAvailableListener, layout, inputSize)
        }
    }
}
/********************************************************************************************************
 * Function necessary for choosing optimal size
 ********************************************************************************************************/
open class CompareSizesByArea: Comparator<Size>{
    override fun compare(p0: Size?, p1: Size?): Int {
        return java.lang.Long.signum(p0!!.width.toLong() * p0!!.height.toLong()
                - p1!!.width.toLong() * p1!!.height.toLong())
    }
}
/********************************************************************************************************
 * Sets session camera callback
 ********************************************************************************************************/
fun interface ConnectionCallback{
    fun onPreviewSizeChosen(size: Size, cameraRotation: Int)
}
