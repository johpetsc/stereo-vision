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
        #img = cv.resize(img, (640, 360))
        img = cv.resize(img, (WIDTH, HEIGHT))
        ret, corners = cv.findChessboardCorners(img, (COLUMNS, ROWS),None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            #print(objp)
            #print(corners)
            img = cv.drawChessboardCorners(img, (COLUMNS,ROWS), corners, ret)
            #cv.imshow('Image',img)
            #cv.waitKey(500000)

    return objpoints, imgpoints, img

def camCalibration(folder):
    objp = np.zeros((ROWS*COLUMNS), np.float32)
    objp[:,:2] = np.mgrid[0:COLUMNS,0:ROWS].T.reshape(-1,1)
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    path = "data/images/" + folder + "/*.jpg"
    images = glob.glob(path)
    objpoints, imgpoints, frame = findBoardInImage(images, objp, objpoints, imgpoints)
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (frame.shape[1], frame.shape[0]),None,None)
    h, w = frame.shape[:1]
    newmtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

    return mtx, dist, newmtx

def extrinsicCalibration(mtx1, dist1, mtx2, dist2):
    objp = np.zeros((8,3), np.float32)
    objp[:,:2] = np.mgrid[0:4,0:1].T.reshape(-1,2)
    objp[0] = [0, 0, 0]
    objp[1] = [0, 1.40, 0]
    objp[2] = [0, 0, 0]
    objp[3] = [2.60, 1.40, 0]
    objp[4] = [0, 0.70, 0]
    objp[5] = [2.60, 0.70, 0]
    objp[6] = [1.30, 0, 0]
    objp[7] = [1.30, 1.40, 0]
    
    objpoints = [[0, 0, 0],
                 [0, 140, 0],
                 [260, 0, 0],
                 [260, 140, 0],
                 [0, 70, 0],
                 [260, 70, 0],
                 [130, 0, 0],
                 [130, 140, 0]]
                 
    imgp = np.zeros((8,2), np.float32)
    imgp[:,:2] = np.mgrid[0:4,0:2].T.reshape(-1,2)

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

    retval1, rvec1, tvec1 = cv.solvePnP(np.float32(objpoints), np.float32(imgpoints1), mtx1, dist1)
    dst1, jacobian1 = cv.Rodrigues(rvec1)
    
    imgp2 = np.zeros((8,2), np.float32)
    imgp2[:,:2] = np.mgrid[0:4,0:2].T.reshape(-1,2)
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
    retval2, rvec2, tvec2 = cv.solvePnP(np.float32(objpoints), np.float32(imgpoints2), mtx2, dist2)
    dst2, jacobian2 = cv.Rodrigues(rvec2)

    X, Y, Z = np.dot(-np.transpose(dst1), tvec1)
    print(X, Y, Z)
    X, Y, Z = np.dot(-np.transpose(dst2), tvec2)
    print(X, Y, Z)
    cameraPosition = -np.matrix(dst1).T * np.matrix(tvec1)
    #print(cameraPosition)
    cameraPosition = -np.matrix(dst2).T * np.matrix(tvec2)
    #print(cameraPosition)

    objpoints = [] # 3d point in real world space
    objpoints.append(objp)
    
    return objpoints, imgpoints1, imgpoints2, dst1, dst2, tvec1, tvec2, rvec1, rvec2

def stereo():
    camera1 = cv.VideoCapture('data/out/camera1/camera1_calibrada.mp4')
    #camera1 = cv.VideoCapture('videos/camera1.mp4')
    camera2 = cv.VideoCapture('data/out/camera2/camera2_calibrada.mp4')
    #camera2 = cv.VideoCapture('videos/camera2.mp4')
    for _ in range(89):
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

def _3DProjection(rvec, tvec, mtx, dist, imgpoints, frame):
    axis = np.float32([[260,120,0], [210,70,0], [260,70,50]]).reshape(-1,3)
    proj, jac = cv.projectPoints(axis, rvec, tvec, mtx, dist)
    corner = tuple(imgpoints[0][5].ravel())
    img = frame
    img = cv.line(img, corner, tuple(proj[0].ravel()), (255, 0, 0), 3)
    img = cv.line(img, corner, tuple(proj[1].ravel()), (0, 255, 0), 3)
    img = cv.line(img, corner, tuple(proj[2].ravel()), (0, 0, 255), 3)
    """img = cv.line(img, corner, tuple(imgpoints_r[0][4].ravel()), (0, 0, 255), 3)
    img = cv.line(img, corner, tuple(imgpoints_r[0][5].ravel()), (0, 255, 0), 3)
    img = cv.line(img, corner, tuple(imgpoints_r[0][6].ravel()), (0, 0, 0), 3)
    img = cv.line(img, corner, tuple(imgpoints_r[0][7].ravel()), (0, 255, 255), 3)"""
    cv.imshow('Projection', img)
    cv.waitKey(888888)
    cv.destroyAllWindows()        

def pls(rmap_r_x, rmap_r_y, rmap_l_x, rmap_l_y):
    camera1 = cv.VideoCapture('data/out/camera1/camera1_calibrada.mp4')
    camera2 = cv.VideoCapture('data/out/camera2/camera2_calibrada.mp4')
    points_r = []
    points_l = []
    for _ in range(137):
        frame_r = camera1.read()[1]
    for _ in range(47):
       frame_l = camera2.read()[1]

    dst_r = cv.remap(frame_r, rmap_r_x, rmap_r_y, interpolation=cv.INTER_LINEAR)
    dst_l = cv.remap(frame_l, rmap_l_x, rmap_l_y, interpolation=cv.INTER_LINEAR)
    dst_r = cv.cvtColor(dst_r, cv.COLOR_BGR2LAB)
    dst_l = cv.cvtColor(dst_l, cv.COLOR_BGR2LAB)
    
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

        dst_r = cv.remap(frame_r, rmap_r_x, rmap_r_y, interpolation=cv.INTER_LINEAR)
        dst_l = cv.remap(frame_l, rmap_l_x, rmap_l_y, interpolation=cv.INTER_LINEAR)
        dst_r = cv.cvtColor(dst_r, cv.COLOR_BGR2LAB)
        dst_l = cv.cvtColor(dst_l, cv.COLOR_BGR2LAB)
        pos_l = tracker_l.update(dst_l)
        pos_r = tracker_r.update(dst_r)
        p1_r = (int(pos_r[1][0]), int(pos_r[1][1]))
        p2_r = (int(pos_r[1][0] + pos_r[1][2]), int(pos_r[1][1] + pos_r[1][3]))
        cv.rectangle(dst_r, p1_r, p2_r, (255, 0, 0))
        p1_l = (int(pos_l[1][0]), int(pos_l[1][1]))
        p2_l = (int(pos_l[1][0] + pos_l[1][2]), int(pos_l[1][1] + pos_l[1][3]))
        cv.rectangle(dst_l, p1_l, p2_l, (255, 0, 0))

        pnt_r[0] = [int(pos_r[1][0]+(pos_r[1][2]/2)), int(pos_r[1][1]+(pos_r[1][3]/2))]
        pnt_l[0] = [int(pos_l[1][0]+(pos_l[1][2]/2)), int(pos_l[1][1]+(pos_l[1][3]/2))]

        points_r.append(pnt_r)
        points_l.append(pnt_l)

        frame = np.hstack((dst_l, dst_r))
        cv.imshow('Tracking', frame)
        if cv.waitKey(50000) & 0xFF == ord('q'):
            if cv.waitKey(50000) & 0xFF == ord('w'):
                while(1):
                    frame_r = camera1.read()[1]
                    frame_l = camera2.read()[1]
                    dst_r = cv.remap(frame_r, rmap_r_x, rmap_r_y, interpolation=cv.INTER_LINEAR)
                    dst_l = cv.remap(frame_l, rmap_l_x, rmap_l_y, interpolation=cv.INTER_LINEAR)
                    frame = np.hstack((dst_l, dst_r))
                    cv.imshow('Tracking', frame)
                    if cv.waitKey(50000) & 0xFF == ord('q'):
                        break
            else:
                break

    return points_r, points_l

def calibration():
    #opt, folder = calibrationMenu()
    size = (1280, 720)
    #Requisito 1
    mtx_r, dist_r, newmtx_r = camCalibration('camera1')
    mtx_l, dist_l, newmtx_l = camCalibration('camera2')

    #Calibracao de instrinsecos pronta
    #Requisito 2
    objpoints, imgpoints_r, imgpoints_l, dst_r, dst_l, tvec_r, tvec_l, rvec_r, rvec_l = extrinsicCalibration(mtx_r, dist_r, mtx_l, dist_l)
    #Calibracao de extrinsecos pronta
    #Requisito 3
    retval, cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, 
                                                                                                    mtx_l, dist_l, mtx_r, 
                                                                                                    dist_r, tuple(size))
    R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2 = cv.stereoRectify(cameraMatrix_l, distCoeffs_l, cameraMatrix_r, distCoeffs_r, tuple(size), R, T, alpha=0, newImageSize=tuple(size))
    
    rmap_r_x, rmap_r_y = cv.initUndistortRectifyMap(mtx_r, dist_r, R_r, P_r, tuple(size), cv.CV_32F)
    rmap_l_x, rmap_l_y = cv.initUndistortRectifyMap(mtx_l, dist_l, R_l, P_l, tuple(size), cv.CV_32F)
    
    frame_r, frame_l = stereo()
    #_3DProjection(rvec_r, tvec_r, mtx_r, dist_r, imgpoints_r, frame_r)
    #_3DProjection(rvec_l, tvec_l, mtx_l, dist_l, imgpoints_l, frame_l)
    cv.destroyAllWindows()
    dst_r = cv.remap(frame_r, rmap_r_x, rmap_r_y, interpolation=cv.INTER_LINEAR)
    dst_l = cv.remap(frame_l, rmap_l_x, rmap_l_y, interpolation=cv.INTER_LINEAR)
    dst_rROI = dst_r[100:500, :1000]
    dst_lROI = dst_l[100:500, 280:]

    """frame = np.hstack((dst_l, dst_r))
    frame = cv.line(frame, (0, 303), (2000,303), (255, 0, 0), 1)
    frame = cv.line(frame, (0, 290), (2000,290), (0, 255, 0), 1)
    frame = cv.line(frame, (0, 131), (2000,131), (0, 0, 255), 1)
    frame = cv.line(frame, (0, 146), (2000,146), (255, 255, 0), 1)
    frame = cv.line(frame, (0, 87), (2000,87), (0, 255, 255), 1)
    frame = cv.line(frame, (0, 75), (2000,75), (255, 0, 255), 1)
    cv.imwrite('data/out/camera1/resultado.jpg', frame)
    cv.imwrite('data/out/camera2/resultado.jpg', dst_l)
    frame1_gray = cv.cvtColor(dst_r, cv.COLOR_BGR2GRAY)
    frame2_gray = cv.cvtColor(dst_l, cv.COLOR_BGR2GRAY)
    cv.imshow('ffa', frame)
    cv.waitKey(888888)
    ster = cv.StereoSGBM_create(minDisparity=11, numDisparities=160, blockSize=11,
                                P1=50, P2=100, disp12MaxDiff=10, uniquenessRatio=12,
                                speckleWindowSize=0, speckleRange=2, preFilterCap=100)
    #ster = cv.StereoBM_create(numDisparities=16, blockSize=11)

    
    disparity = ster.compute(frame2_gray, frame1_gray)
    #disparity = disparity[:, 400:]
    image3D = cv.reprojectImageTo3D(disparity, Q)
    depth = image3D[:,:,1]
    norm = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    
    cv.imshow('aaa', norm)
    #cv.imwrite('data/out/camera1/disparidade.jpg', norm)
    cv.waitKey(888888)
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='10%', pad=0.05)
    im = ax.imshow(norm, 'gray')
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()"""

    a, b = pls(rmap_r_x, rmap_r_y, rmap_l_x, rmap_l_y)
    pontos = cv.triangulatePoints(P_l, P_r, np.float32(b), np.float32(a))
    X, Y, Z = pontos[0]/pontos[3], pontos[1]/pontos[3], pontos[2]/pontos[3]
   
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle("Variação do objeto para os eixos X, Y, Z")

    ax1.plot(X)
    ax1.set_ylabel('X')

    ax2.plot(Y)
    ax2.set_ylabel('Y')

    ax3.plot(Z)
    ax3.set_ylabel('Z')
    plt.show()

def main():
    if sys.argv[1] == 'c':
        calibration()
        #cal.undistort(5, 'camera1')
    else:
        stereo()

if __name__ == "__main__":
    main()
