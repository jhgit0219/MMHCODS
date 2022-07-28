# MMHCODS
Mobile Motorcycle Helmet and Counterflowing/Overtaking Detection System

This System is an android deployment system of custom trained AI models that detects motorcycles, helmets, license plates, and lanes. 
These results are then used to create a violation detection system.

# TODO
- Screen Scaling Issue

There seems to be a problem with regards to the scaling of the coordinates to the screen. 

The pertinent files are:
  - AutoAdjustView
  - OverlayView
  - DetectorActivity
  - YoloV5Classifier
  - LaneClassifier

Unsure if the problem lies between the scaling of the conversion of the detected coordinates to the canvas, or from the canvas to the screen. The conversion of the coordinates from the detection to the canvas is based on the ImageUtils.convertToRange.

# How-To

First, clone the repository.

Next step, download this [file](https://drive.google.com/file/d/1UQpXrN18b_sVGG8z0BtRuNbWqtEkkwf9/view?usp=sharing "OBB File") and place it in your android device at Android > obb > com.anlehu.mmhcods . Make this folder if it does not exist, or run the cloned android studio project so the folder gets created.

Next, there should be some dependency issues with the gradle files. Simply download opencv for android and follow the steps listed in their website. The dependency link should be :opencv, not :sdk. 

Fix the other gradle issues and you should be good to go.

Contact me for help if you get stuck somewhere.
