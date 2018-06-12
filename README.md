# OpenCV Demo
A list of OpenCV demos


## Stitch
 A demo demonstrates how to use feature detector, find transform matrix , warp img, and stitch together.
   
 usage: ./stitch img0.jpg img1.jpg img2.jpg..., note img should be ordered from left to right.
 ![stitch result0](doc/stitch_result0.jpg)
 ![stitch result1](doc/stitch_result1.jpg)

##  SiftGPUStitch
 A demo just like Stitch, but use [SiftGPU](https://github.com/pitzer/SiftGPU.git) to extract features. SiftGPU is A GPU implementation of David Lowe's Scale Invariant Feature Transform, writeen by Changchang wu. 