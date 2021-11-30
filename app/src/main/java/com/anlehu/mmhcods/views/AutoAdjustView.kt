package com.anlehu.mmhcods.views

import android.content.Context
import android.util.AttributeSet
import android.view.TextureView
import java.lang.IllegalArgumentException

class AutoAdjustView: TextureView {
    var widthRatio = 0
    var heightRatio = 0

    constructor(context: Context): this(context, null)
    constructor(context: Context, attributeSet: AttributeSet?) : this(context, attributeSet!!, 0)
    constructor(context: Context, attributeSet: AttributeSet, defStyle: Int) : super(context, attributeSet, defStyle)

    fun setAspectRatio(width: Int, height: Int){
        if(width < 0 || height < 0){
            throw IllegalArgumentException("Size below 0")
        }
        widthRatio = width
        heightRatio = height
        requestLayout()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec)
        var width = MeasureSpec.getSize(widthMeasureSpec)
        var height = MeasureSpec.getSize(heightMeasureSpec)
        if(widthRatio == 0 || heightRatio == 0){
            setMeasuredDimension(width, height)
        }else{
            if(width < (height * widthRatio / heightRatio)){
                setMeasuredDimension(width, width * heightRatio / widthRatio)
            }else{
                setMeasuredDimension(height *  widthRatio / heightRatio, height)
            }
        }
    }
}