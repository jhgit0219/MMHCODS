package com.anlehu.mmhcods.views

import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View
import java.util.*

class OverlayView: View {
    var callbacks: MutableList<DrawCallback> = LinkedList()

    constructor(context: Context, attributeSet: AttributeSet) : super(context, attributeSet)
    fun addCallback(callback: DrawCallback){callbacks.add(callback)}

    override fun draw(canvas: Canvas?) {
        for(callback in callbacks){
            callback.drawCallback(canvas!!)
        }
        super.draw(canvas)
    }
    interface DrawCallback{
        fun drawCallback(canvas: Canvas)
    }
}
