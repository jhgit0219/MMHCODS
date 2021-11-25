package com.anlehu.mmhcods.utils

import android.graphics.*
import android.graphics.Paint.Align
import android.util.Log
import java.util.*

class BorderedText {
    private var inPaint: Paint = Paint()
    private var exPaint: Paint = Paint()

    private var textSize: Float = 0f

    constructor(textSize: Float){
        BorderedText(Color.WHITE, Color.BLACK, textSize)
        Log.d("SIZE_TEXT", "$textSize")

    }

    constructor(inColor: Int, exColor: Int, textSize: Float){
        inPaint = Paint()
        inPaint.textSize = textSize
        inPaint.color = inColor
        inPaint.style = Paint.Style.FILL
        inPaint.isAntiAlias = false
        inPaint.alpha = 255

        exPaint = Paint()
        exPaint.textSize = textSize
        exPaint.color = exColor
        exPaint.style = Paint.Style.FILL_AND_STROKE
        exPaint.strokeWidth = textSize / 8
        exPaint.isAntiAlias = false
        exPaint.alpha = 255

        this.textSize = textSize

    }

    fun setTypeFace(typeFace: Typeface){
        inPaint.typeface = typeFace
        exPaint.typeface = typeFace

    }

    fun drawText(canvas: Canvas, posX: Float, posY: Float, text: String?) {
        canvas.drawText(text!!, posX, posY, exPaint)
        canvas.drawText(text, posX, posY, inPaint)
    }

    fun drawText(
        canvas: Canvas, posX: Float, posY: Float, text: String?, bgPaint: Paint?
    ) {
        val width: Float = exPaint.measureText(text)
        val textSize: Float = exPaint.textSize
        val paint = Paint(bgPaint)
        paint.style = Paint.Style.FILL
        paint.alpha = 160
        canvas.drawRect(posX, posY + textSize.toInt(), posX + width.toInt(), posY, paint)
        canvas.drawText(text!!, posX, posY + textSize, inPaint)
    }

    fun drawLines(canvas: Canvas?, posX: Float, posY: Float, lines: Vector<String?>) {
        var lineNum = 0
        for (line in lines) {
            drawText(canvas!!, posX, posY - getTextSize() * (lines.size - lineNum - 1), line)
            ++lineNum
        }
    }

    fun setInteriorColor(color: Int) {
        inPaint.color = color
    }

    fun setExteriorColor(color: Int) {
        exPaint.color = color
    }

    fun getTextSize(): Float {
        return textSize
    }

    fun setAlpha(alpha: Int) {
        inPaint.alpha = alpha
        exPaint.alpha = alpha
    }

    fun getTextBounds(
        line: String?, index: Int, count: Int, lineBounds: Rect?
    ) {
        inPaint.getTextBounds(line, index, count, lineBounds)
    }

    fun setTextAlign(align: Align?) {
        inPaint.textAlign = align
        exPaint.textAlign = align
    }
}