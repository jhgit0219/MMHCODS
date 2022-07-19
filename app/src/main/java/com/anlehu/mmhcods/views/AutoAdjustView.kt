package com.anlehu.mmhcods.views

import android.content.Context
import android.util.AttributeSet
import android.util.Log
import android.view.TextureView
import android.view.ViewGroup.MarginLayoutParams

class AutoAdjustView: TextureView {
    var widthRatio = 0
    var heightRatio = 0
    var keepAspect = false

    constructor(context: Context): this(context, null)
    constructor(context: Context, attributeSet: AttributeSet?) : this(context, attributeSet!!, 0)
    constructor(context: Context, attributeSet: AttributeSet, defStyle: Int) : super(context, attributeSet, defStyle)

    fun setAspectRatio(width: Int, height: Int, bool: Boolean){
        if(width < 0 || height < 0){
            throw IllegalArgumentException("Size below 0")
        }
        Log.d("ASPECT", "$width x $height")
        widthRatio = width
        heightRatio = height
        keepAspect = bool
        requestLayout()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec)
        val width = MeasureSpec.getSize(widthMeasureSpec)
        val height = MeasureSpec.getSize(heightMeasureSpec)
        val margin = (height - width) / 2

        if(keepAspect){
            keepAspect = false
            val margins = MarginLayoutParams::class.java.cast(layoutParams)
            margins!!.topMargin = 0
            margins.bottomMargin = 0
            margins.leftMargin = -margin
            margins.rightMargin = -margin
            layoutParams = margins
        }
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