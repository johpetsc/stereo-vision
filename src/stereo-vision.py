import sys
import cv2 as cv
import numpy as np
import glob

ROWS = 8
COLUMNS = 6
HEIGHT = 720
WIDTH = 1280

def findBoardInImage(images, objp, objpoints, imgpoints):
    for image in images:
        img = cv.imread(image)
        img = cv.resize(img, (WIDTH, HEIGHT))
        ret, corners = cv.findChessboardCorners(img, (COLUMNS, ROWS),None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            img = cv.drawChessboardCorners(img, (COLUMNS,ROWS), corners, ret)
            #cv.imshow('Image',img)
            #cv.waitKey(500000)

    return objpoints, imgpoints, img

def camCalibration(folder): # intrinsic calibration
    objp = np.zeros((ROWS*COLUMNS, 3), np.float32)
    objp[:,:2] = np.mgrid[0:COLUMNS,0:ROWS].T.reshape(-1,2)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    path = "data/images/" + folder + "/*.jpg" # reads every image in the folder
    images = glob.glob(path)
    # finds board coordinates
    objpoints, imgpoints, frame = findBoardInImage(images, objp, objpoints, imgpoints)
    cv.destroyAllWindows()
    # calibrates the camera 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (frame.shape[1], frame.shape[0]),None,None)
    h, w = frame.shape[:2]
    newmtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

    return mtx, dist, newmtx

def extrinsicCalibration(mtx1, dist1, mtx2, dist2):
    # 3d space coordinates
    objp = np.zeros((8,3), np.float32)
    objp[0] = [0, 0, 0]
    objp[1] = [0, 1.40, 0]
    objp[2] = [2.60, 0, 0]
    objp[3] = [2.60, 1.40, 0]
    objp[4] = [0, 0.70, 0]
    objp[5] = [2.60, 0.70, 0]
    objp[6] = [1.30, 0, 0]
    objp[7] = [1.30, 1.40, 0]
    
    objpoints = [[0, 0, 0],
                 [0, 1.40, 0],
                 [2.60, 0, 0],
                 [2.60, 1.40, 0],
                 [0, 0.70, 0],
                 [2.60, 0.70, 0],
                 [1.30, 0, 0],
                 [1.30, 1.40, 0]]
    # 2d points in camera 1           
    imgp = np.zeros((8,2), np.float32)
    imgp[0] = [716, 60]
    imgp[1] = [1072, 110]
    imgp[2] = [210, 395]
    imgp[3] = [970, 660]
    imgp[4] = [880, 83]
    imgp[5] = [520, 505]
    imgp[6] = [557, 165]
    imgp[7] = [1045, 261]
    imgpoints1 = [] # 3d point in real world space
    imgpoints1.append(imgp)
    # generates the extrinsic values for camera 1
    retval1, rvec1, tvec1 = cv.solvePnP(np.float32(objpoints), np.float32(imgpoints1), mtx1, dist1)
    dst1, jacobian1 = cv.Rodrigues(rvec1)
    # 2d points in camera 2
    imgp2 = np.zeros((8,2), np.float32)
    imgp2[0] = [220, 50]
    imgp2[1] = [630, 34]
    imgp2[2] = [50, 670]
    imgp2[3] = [1083, 487]
    imgp2[4] = [434, 42]
    imgp2[5] = [640, 570]
    imgp2[6] = [175, 212]
    imgp2[7] = [760, 170]
    imgpoints2 = [] # 3d point in real world space
    imgpoints2.append(imgp2)
    # generates the extrinsic values for camera 2
    retval2, rvec2, tvec2 = cv.solvePnP(np.float32(objpoints), np.float32(imgpoints2), mtx2, dist2)
    dst2, jacobian2 = cv.Rodrigues(rvec2)

    # calculates the camera position in the 3d space
    cameraPosition = -np.matrix(dst1).T * np.matrix(tvec1)
    #print(cameraPosition)
    cameraPosition = -np.matrix(dst2).T * np.matrix(tvec2)
    #print(cameraPosition)

    objpoints = [] # 3d point in real world space
    objpoints.append(objp)
    
    return objpoints, imgpoints1, imgpoints2, dst1, dst2, tvec1, tvec2, rvec1, rvec2

def stereo(): # synchronizes different camera frames
    camera1 = cv.VideoCapture('data/out/camera1/camera1_20fps.mp4')
    camera2 = cv.VideoCapture('data/out/camera2/camera2_20fps.mp4')
    for _ in range(102): # ignores initial camera 1 frames
        frame1 = camera1.read()[1]
    while(1):
        frame1 = camera1.read()[1]
        frame2 = camera2.read()[1]
        frame1 = cv.resize(frame1, (1280, 720))
        frame2 = cv.resize(frame2, (1280, 720))
        frame1_resized = cv.resize(frame1, (640, 360))
        frame2_resized = cv.resize(frame2, (640, 360))
        frame = np.hstack((frame2_resized, frame1_resized))
        cv.imshow('Camera 2                                                                                        Camera 1', frame)
        
        if cv.waitKey(50000) & 0xFF == ord('q'):
            break
    return frame1, frame2

def _3DProjection(rvec, tvec, mtx, dist, imgpoints, frame): # generates the X, Y and Z axis for an image frame
    axis = np.float32([[2.60,1.20,0], [2.10,0.70,0], [2.60,0.70,0.50]]).reshape(-1,3)
    proj, jac = cv.projectPoints(axis, rvec, tvec, mtx, dist)
    corner = tuple(imgpoints[0][5].ravel())
    img = frame
    img = cv.line(img, corner, tuple(proj[0].ravel()), (255, 0, 0), 3)
    img = cv.line(img, corner, tuple(proj[1].ravel()), (0, 255, 0), 3)
    img = cv.line(img, corner, tuple(proj[2].ravel()), (0, 0, 255), 3)
    cv.imshow('Projection', img)
    cv.imwrite('data/results/axis.jpg', img)
    cv.waitKey(888888)
    cv.destroyAllWindows()        

def tracking(rmap_r_x, rmap_r_y, rmap_l_x, rmap_l_y): # 3d object tracking in both cameras, storing the 2d coordinates for each frame
    camera1 = cv.VideoCapture('data/out/camera1/camera1_20fps.mp4')
    camera2 = cv.VideoCapture('data/out/camera2/camera2_20fps.mp4')
    points_r = [] # 2d points for camera 1 frames
    points_l = [] # 2d points for camera 2 frames
    for _ in range(150): # skips the frames preceding the car hitting the board
        frame_r = camera1.read()[1]
    for _ in range(47):
       frame_l = camera2.read()[1]

    frame_r = cv.resize(frame_r, (1280, 720))
    frame_l = cv.resize(frame_l, (1280, 720))
    dst_r = cv.remap(frame_r, rmap_r_x, rmap_r_y, interpolation=cv.INTER_LINEAR)
    dst_l = cv.remap(frame_l, rmap_l_x, rmap_l_y, interpolation=cv.INTER_LINEAR)
    ROI_r = cv.selectROI("Tracking >", dst_r)
    tracker_r = cv.TrackerMIL_create()
    ROI_l = cv.selectROI("Tracking <", dst_l)
    tracker_l = cv.TrackerMIL_create()
    pos_l = tracker_l.init(dst_l, ROI_l)
    pos_r = tracker_r.init(dst_r, ROI_r)
    pos_l = tracker_l.update(dst_l)
    pos_r = tracker_r.update(dst_r)

    cv.destroyAllWindows()
    

    while camera1.isOpened():
        pnt_r = np.zeros((1,2), np.float32)
        pnt_l = np.zeros((1,2), np.float32)
        frame_r = camera1.read()[1]
        frame_l = camera2.read()[1]

        frame_r = cv.resize(frame_r, (1280, 720))
        frame_l = cv.resize(frame_l, (1280, 720))
        dst_r = cv.remap(frame_r, rmap_r_x, rmap_r_y, interpolation=cv.INTER_LINEAR)
        dst_l = cv.remap(frame_l, rmap_l_x, rmap_l_y, interpolation=cv.INTER_LINEAR)
        pos_l = tracker_l.update(dst_l)
        pos_r = tracker_r.update(dst_r)
        # generates the bbox
        p1_r = (int(pos_r[1][0]), int(pos_r[1][1]))
        p2_r = (int(pos_r[1][0] + pos_r[1][2]), int(pos_r[1][1] + pos_r[1][3]))
        cv.rectangle(dst_r, p1_r, p2_r, (255, 0, 0))
        p1_l = (int(pos_l[1][0]), int(pos_l[1][1]))
        p2_l = (int(pos_l[1][0] + pos_l[1][2]), int(pos_l[1][1] + pos_l[1][3]))
        cv.rectangle(dst_l, p1_l, p2_l, (255, 0, 0))
        # 2d coordinate
        pnt_r[0] = [int(pos_r[1][0]+(pos_r[1][2]/2)), int(pos_r[1][1]+(pos_r[1][3]/2))]
        pnt_l[0] = [int(pos_l[1][0]+(pos_l[1][2]/2)), int(pos_l[1][1]+(pos_l[1][3]/2))]
        points_r.append(pnt_r)
        points_l.append(pnt_l)

        frame = np.hstack((dst_l, dst_r))
        cv.imshow('Tracking', frame)
        if cv.waitKey(50000) & 0xFF == ord('q'): # Q->Q: break, Q->W: pause tracking
            if cv.waitKey(50000) & 0xFF == ord('w'): # ignores the frames where te object leaves the frame
                while(1): 
                    frame_r = camera1.read()[1]
                    frame_l = camera2.read()[1]
                    frame_r = cv.resize(frame_r, (1280, 720))
                    frame_l = cv.resize(frame_l, (1280, 720))
                    points_r.append(pnt_r)
                    points_l.append(pnt_l)
                    dst_r = cv.remap(frame_r, rmap_r_x, rmap_r_y, interpolation=cv.INTER_LINEAR)
                    dst_l = cv.remap(frame_l, rmap_l_x, rmap_l_y, interpolation=cv.INTER_LINEAR)
                    frame = np.hstack((dst_l, dst_r))
                    cv.imshow('Tracking', frame)
                    if cv.waitKey(50000) & 0xFF == ord('q'):
                        break
            else:
                break

    return points_r, points_l

def stereo_vision(req):
    print('...')
    size = (WIDTH, HEIGHT)
    mtx_r, dist_r, newmtx_r = camCalibration('camera1')
    mtx_l, dist_l, newmtx_l = camCalibration('camera2')

    if req == 1:
        print("CAMERA 1")
        print("Intrinsic Matrix:")
        print(mtx_r)
        print("Distortion:")
        print(dist_r)
        print("CAMERA 2")
        print("Intrinsic Matrix:")
        print(mtx_l)
        print("Distortion:")
        print(dist_l)

    objpoints, imgpoints_r, imgpoints_l, dst_r, dst_l, tvec_r, tvec_l, rvec_r, rvec_l = extrinsicCalibration(mtx_r, dist_r, mtx_l, dist_l)
    
    if req == 2:
        print("Rotation Matrix:")
        print(dst_r)
        print("Translation Vector:")
        print(tvec_r)
        print("Camera position:")
        print(np.transpose(dst_r))
        print(-np.transpose(dst_r))
        X, Y, Z = np.dot(-np.transpose(dst_r), tvec_r)
        print("X:", X, "Y:", Y, "Z:", Z)
        print("Rotation Matrix:")
        print(dst_l)
        print("Translation Vector:")
        print(tvec_l)
        print("Camera position:")
        X, Y, Z = np.dot(-np.transpose(dst_l), tvec_l)
        print("X:", X, "Y:", Y, "Z:", Z)

    if req > 2:
        # stereo calibration
        retval, cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, 
                                                                                                        mtx_l, dist_l, mtx_r, 
                                                                                                        dist_r, tuple(size))

        R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2 = cv.stereoRectify(cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, tuple(size), R, T, alpha=0.92, newImageSize=tuple(size))
        
        rmap_r_x, rmap_r_y = cv.initUndistortRectifyMap(mtx_r, dist_r, R_r, P_r, tuple(size), cv.CV_32F)
        rmap_l_x, rmap_l_y = cv.initUndistortRectifyMap(mtx_l, dist_l, R_l, P_l, tuple(size), cv.CV_32F)
    
    if req == 3:
        frame_r, frame_l = stereo()
        #_3DProjection(rvec_r, tvec_r, mtx_r, dist_r, imgpoints_r, frame_r)
        #_3DProjection(rvec_l, tvec_l, mtx_l, dist_l, imgpoints_l, frame_l)
        cv.destroyAllWindows()
        dst_r = cv.remap(frame_r, rmap_r_x, rmap_r_y, interpolation=cv.INTER_LINEAR)
        dst_l = cv.remap(frame_l, rmap_l_x, rmap_l_y, interpolation=cv.INTER_LINEAR)

        frame = np.hstack((dst_l, dst_r))
        # epipolar lines
        frame = cv.line(frame, (0, 505), (2560,505), (255, 0, 0), 1)
        frame = cv.line(frame, (0, 480), (2560,480), (0, 255, 0), 1)
        frame = cv.line(frame, (0, 160), (2560,160), (0, 0, 255), 1)
        frame = cv.line(frame, (0, 185), (2560,185), (255, 255, 0), 1)
        frame = cv.line(frame, (0, 67), (2560,67), (0, 255, 255), 1)
        frame = cv.line(frame, (0, 45), (2560,45), (255, 0, 255), 1)

        cv.imwrite('data/out/camera1/resultado.jpg', frame)
        frame1_gray = cv.cvtColor(dst_r, cv.COLOR_BGR2GRAY)
        frame2_gray = cv.cvtColor(dst_l, cv.COLOR_BGR2GRAY)
        cv.imshow('Rectified', frame)
        cv.waitKey(888888)
        ster = cv.StereoSGBM_create(minDisparity=11, numDisparities=16, blockSize=11,
                                    P1=50, P2=100, disp12MaxDiff=10, uniquenessRatio=12,
                                    speckleWindowSize=0, speckleRange=2, preFilterCap=100)
        
        disparity = ster.compute(frame2_gray, frame1_gray) # computes the disparity
        image3D = cv.reprojectImageTo3D(disparity, Q) 
        depth = image3D[:,:,1] # depth coordinates
        norm = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        depth = cv.normalize(depth, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        divider = make_axes_locatable(ax)
        divider2 = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='10%', pad=0.05)
        cax2 = divider2.append_axes('right', size='10%', pad=0.05)
        im = ax.imshow(norm, 'gray')
        im2 = ax2.imshow(depth, 'gray')
        fig.colorbar(im, cax=cax, orientation='vertical')
        fig.suptitle("Disparity")
        fig2.colorbar(im2, cax=cax2, orientation='vertical')
        fig2.suptitle("Depth")
        plt.show()
    
    if req == 4:
        points_l, points_r = tracking(rmap_r_x, rmap_r_y, rmap_l_x, rmap_l_y)
        points = cv.triangulatePoints(P_l, P_r, np.float32(points_l), np.float32(points_r))
        X, Y, Z = points[0]/points[3], points[1]/points[3], points[2]/points[3]# X: x/w, Y: y/w, Z: z/w

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable1
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.suptitle("Object variation for axis X, Y, Z")

        ax1.plot(X)
        ax1.set_ylabel('X')
        ax1.set_ylim([0.5, 1])

        ax2.plot(Y)
        ax2.set_ylabel('Y')
        ax2.set_ylim([-1, 2])

        ax3.plot(Z)
        ax3.set_ylabel('Z')
        ax3.set_ylim([-4, -1])
        plt.show()

def camera1_fps(folder): # creates a new video file for camera 1 to standardize its fps
    video = cv.VideoCapture("data/" + folder + ".webm")
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    skip = 0
    frames = 16
    writer = cv.VideoWriter('data/out/' + folder + '/' + folder + '_20fps.mp4', fourcc, 20.0, (1920, 1080))
    while video.isOpened():
        ret, frame = video.read()
        skip += 1
        if ret == False:
            break
        if skip == frames:
            skip = 0  
        else:
            writer.write(frame)
    writer.release()
    video.release()

def camera2_fps(folder): # creates a new video file for camera 2 to standardize its fps
    video = cv.VideoCapture("data/" + folder + ".webm")
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter('data/out/' + folder + '/' + folder + '_20fps.mp4', fourcc, 20.0, (1280, 720))
    while video.isOpened():
        ret, frame = video.read()
        if ret == False:
            break
        writer.write(frame)
        writer.write(frame)
    writer.release()
    video.release()

def main():
    if sys.argv[1] == '--r1':
        stereo_vision(1)
    elif sys.argv[1] == '--r2':
        stereo_vision(2)
    elif sys.argv[1] == '--r3':
        stereo_vision(3)
    elif sys.argv[1] == '--r4':
        stereo_vision(4)
    else: # this is needed for the code to work properly
          # new video files with 20 fps will be creates for both camera 1 and 2
        #22(ish)fps -> 20 fps
        camera1_fps('camera1')
        #10fps -> 20 fps
        camera2_fps('camera2')

if __name__ == "__main__":
    main()
