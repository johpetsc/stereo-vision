# Stereo Vision
Measurements and tracking based on two cameras with different configurations. The program first calibrates both cameras based on pictures of a chess board, calculating the intrinsic matrix and the distortion. After that, based on recordings of two cameras capturing the same scene, the program:

- generates the rotation matrix and translation vector
- calculates both cameras position in a 3D space
- executes the stereo calibration
- rectifies and synchronizes the frames of the cameras
- generates epipolar lines for the captured objects
- computes the disparity and depth
- tracks a moving object and calculates its position in a 3d space

# Requirements
```
    $ pip install opencv-python
```

# Execute
Camera intrinsic calibration:
```
    $ python src/stereo-vision.py --r1
```
Camera extrinsic calibration:
```
    $ python src/stereo-vision.py --r2
```
Cameras disparity and depth:
```
    $ python src/stereo-vision.py --r3
```
Object tracking:
```
    $ python src/stereo-vision.py --r4
```
1. Select an object to be tracked in both camera 1 and then press ENTER
2. Select an object to be tracked in both camera 2 and then press ENTER
3. Press Q and then hold W to track the object (release W to pause)
4. Press Q twice to exit before the video finishes