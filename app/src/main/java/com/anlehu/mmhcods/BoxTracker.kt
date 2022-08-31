package com.anlehu.mmhcods

import android.content.Context
import android.graphics.*
import android.text.TextUtils
import android.util.Log
import android.util.Pair
import android.util.TypedValue
import com.anlehu.mmhcods.utils.BorderedText
import com.anlehu.mmhcods.utils.Detector.Detection
import java.util.*

class BoxTracker {

    private var trackedObjects: MutableList<TrackedPrediction> = LinkedList()
    private var borderedText: BorderedText
    private var textSize: Float
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

    constructor(context: Context){
        boxPaint.color = Color.RED
        boxPaint.strokeWidth = 10.0f
        boxPaint.strokeCap = Paint.Cap.ROUND
        boxPaint.strokeJoin = Paint.Join.ROUND
        boxPaint.strokeMiter = 100f
        boxPaint.style = Paint.Style.STROKE

        textSize = TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            TEXT_SIZE_DIP,
            context.resources.displayMetrics
        )
        borderedText = BorderedText(textSize)

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
    fun drawDebugBoxes(canvas: Canvas){
        val textPaint = Paint()
        textPaint.color = Color.WHITE
        textPaint.textSize = 60.0f

        val boxPaint = Paint()
        boxPaint.color = Color.RED
        boxPaint.alpha = 200
        boxPaint.style = Paint.Style.STROKE

        for (detection in screenRects) {
            val rect = detection.second
            canvas.drawRect(rect, boxPaint)
            canvas.drawText("" + detection.first, rect.left, rect.top, textPaint)
            borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first)
        }
    }

    @Synchronized
    fun draw(canvas: Canvas) {

        for (recognition in trackedObjects) {
            if(recognition.id == 1){
                boxPaint.color = Color.YELLOW
                Log.d("LANE_DET", "Lane = ${recognition.lanePoints.joinToString(",")}")
                canvas.drawLines(recognition.lanePoints, boxPaint)
            }else{
                val trackedPos = RectF(recognition.location)
                //val trackedPos = RectF(0f, 0f, 1920f, 1080f)
                boxPaint.color = recognition.color
                val cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f
                canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint)
                val labelString = if (!TextUtils.isEmpty(recognition.title)) String.format(
                    "%s %.2f", recognition.title,
                    100 * recognition.detectionConfidence
                ) else String.format("%.2f", 100 * recognition.detectionConfidence)
                //            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.top,
                // labelString);
                borderedText.drawText(
                    canvas, trackedPos.left + cornerSize, trackedPos.top, "$labelString%", boxPaint
                )
            }

        }
        //canvas.drawLines(lanePoints, boxPaint)
    }

    @Synchronized
    fun trackResults(results: List<Detection>, lanes: List<FloatArray>, timeStamp: Long){
        Log.i("BoxTracker:", "${results.size} results from $timeStamp")
        // rectangles to track
        val rectsToTrack: MutableList<Pair<Float, Detection>> = LinkedList()

        // clear current drawings on screen
        screenRects.clear()

        // grab values based on frame to canvas matrix conversion
        var rgbFrameToScreen = Matrix(frameToCanvasMat)

        for(result in results){

            val detFrameRect = RectF(result.location)
            val detScreenRect = RectF()
            rgbFrameToScreen.mapRect(detScreenRect, detFrameRect)
            screenRects.add(Pair<Float, RectF>(result.confidence, detScreenRect))
            rectsToTrack.add(Pair<Float, Detection>(result.confidence, result))

            if(detFrameRect.width() < MIN_SIZE ||detFrameRect.height() < MIN_SIZE){
                continue
            }

            rectsToTrack.add(Pair<Float, Detection>(result.confidence, result))
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

    companion object{
        const val TEXT_SIZE_DIP = 18.0f
        const val MIN_SIZE = 16.0f
    }

}
