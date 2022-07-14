package com.anlehu.mmhcods.views

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet

class TestImageView : androidx.appcompat.widget.AppCompatImageView {

    private val boxPaint = Paint()
    var points: FloatArray? = null
    var points2: FloatArray? = null

    init{
        boxPaint.color = Color.YELLOW
        boxPaint.alpha = 200
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = 3.0f
    }

    constructor(context: Context?) : super(context!!) {
        invalidate()

    }

    constructor(context: Context?, attrs: AttributeSet?) : super(context!!, attrs) {
        invalidate()
    }

    constructor(context: Context?, attrs: AttributeSet?, defStyle: Int) : super(context!!, attrs, defStyle) {
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if(points != null)
            canvas.drawLines(points!!, boxPaint)
        if(points2 != null)
            canvas.drawLines(points2!!, boxPaint)
        //canvas.drawLine(0f, 0f, 720f, 1630f, boxPaint)
    }

    fun drawLane(points: FloatArray, points2: FloatArray){
        this.points = points
        this.points2 = points2
        invalidate()
    }
}