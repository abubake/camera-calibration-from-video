import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from datetime import date
from PIL import Image

'''README:
Usage: determine projection error, intrinsic matrix K, and distortion parameters from a video of a checkerboard pattern with
square of known size.

EX Usage:
projectionError, K, distortion = chessBoardCalib(video_path="C:/Users/baker/anaconda3/envs/NeRF_mapping/2022-10-21_16.56.19.mkv",
 save_directory="C:/Users/baker/anaconda3/envs/NeRF_mapping/data/images/fromROV",
  save_folder_path='data/images/fromROV/2022-10-21_16',
  frameSize = [1920,1080], chessboardSize = [9,7], extraction_rate_s = 60)
  
  "import camera_calibrate as cc"
  
  Inputs:
    - video_path: requires full or partial path to the video. Tested with .mkv and .mp4
        Note: you only need relative path
    - save_directory: folder where you want the images to go
    - save_folder: creates a folder of same name as your video with the frames inside. Make it within the save_folder
        Note: destination folder name at end of save_folder path MUST be same as the video file's name. Otherwise error.
    - frameSize: width, height. Think width is x axis, height is y axis. Starts in top left corner of image.
    - chessBoardsize: dimensions of INNER vertexes. reported as y,x (so height-wise first)
    - extraction rate: 60 is 1 second, meaning it extracts every second
    - chekerboard_squares_mm: size in mm of the checkerboard squares
  Outputs:
    - projectionError
    - K: intrinsic matrix 3x3
    - distortion parameter (1x5), first 2 items are the radial distortion
    
    Areas for Improvement:
    - FIXME: should determine the camera width and length automatically, so it can be eliminated as a parameter
    - FIXME: current setup of multiple filepaths is too confusing for the workflow. Should simply have the directory of the video,
    and the folder you would like the images to go.
    - FIXME: Resolve issue of needing to delete all the images in the directory before running again. This is annoying
    - FIXME: create a new function that calibrates parameters assuming you already have a folder of images
  '''

def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")

def save_frame(video_path, save_dir, gap=60):
    name = video_path.split("/")[-1].split(".")[0]
    save_path = os.path.join(save_dir, name)
    create_dir(save_path)

    cap = cv2.VideoCapture(video_path)
    idx = 0
    
    img_count = 0
    
    # Set time stamp initial values #########################################################
    
    YYYYMMDD = date.today() # 20221014 # Set the year, month, and day here
    hh = 0 # Set the starting hour here (0 to 23)
    mm = 0 # Set the starting minute here
    ss = 0 # Set the starting second here
    ddd = 0 # Set the starting 100th of a second here, please enter in increments of 100
    # EX.) 100, or 200, 300, etc. up to 900
    
    # Every frame extracted is a 1/10 of a second. Code is designed to only handle this rate.
    # Future work could enable any extraction rate desired
    # Generating correct time steps depends on the gap variable, paired with the minimum value
    # ddd imcrements at, which is "+100" each 6 frames for a frame rate of 60 fps
    
    #########################################################################################
    
    if hh < 10:
        hh_string = "0" + str(hh)
    else:
        hh_string = str(hh)           
    if mm < 10:
        mm_string = "0" + str(mm)
    else:
        mm_string = str(mm)             
    if ss < 10:
        ss_string = "0" + str(ss)
    else:
        ss_string = str(ss)           
    if ddd == 0:
        ddd_string = "00" + str(ddd)
    else:
        ddd_string = str(ddd)
                
    timeStamp = hh_string + mm_string + ss_string + ddd_string
    frame_name = "M116" + "_" + str(YYYYMMDD) + "_" + timeStamp + "_" + "img" + str(idx)

    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break
        
        if idx == 0:
            cv2.imwrite(f"{save_path}/{frame_name}.jpg", frame)
            img_count += 1
            
        else:
            
            if idx % gap == 0:
                
                ddd += 1000
                
                if ddd == 1000:
                    ddd = 0
                    ss += 1
                    if ss == 60:
                        ss = 0
                        mm += 1
                        if mm == 60:
                            mm = 0
                            hh += 1
                            if hh == 24:
                                hh = 0
                
                if hh < 10:
                    hh_string = "0" + str(hh)
                else:
                    hh_string = str(hh)
                    
                if mm < 10:
                    mm_string = "0" + str(mm)
                else:
                    mm_string = str(mm)   
                    
                if ss < 10:
                    ss_string = "0" + str(ss)
                else:
                    ss_string = str(ss) 
                    
                if ddd == 0:
                    ddd_string = "00" + str(ddd)
                else:
                    ddd_string = str(ddd)  
                    
                timeStamp = hh_string + mm_string + ss_string + ddd_string
                frame_name = "M116" + "_" + str(YYYYMMDD) + "_" + timeStamp + "_" + "img" + str(idx)   
                cv2.imwrite(f"{save_path}/{frame_name}.jpg", frame)
                img_count += 1

        idx += 1
    return img_count

def chessBoardCalib(video_path="C:/Users/baker/anaconda3/envs/NeRF_mapping/2022-10-21_16.56.19.mkv",
 save_directory="C:/Users/baker/anaconda3/envs/NeRF_mapping/data/images/fromROV",
  save_folder_path='data/images/fromROV/2022-10-21_16',
  frameSize = [1920,1080], chessboardSize = [9,7], extraction_rate_s = 60, checkerboard_squares_mm = 20):
    
    '''
    @FIXES:
    - Fixme: running on different days causes images to stack up in same file bc images get current day file name
    - not a big problem bc this should not have to be done often since only for calibration
    - Fixme: currently depends on a local library, video_to_timestamped_frame. We want to get rid of this as a dependency
    @Params

    @Returns

    '''
    image_count = 0

    video_paths = glob(video_path)
    save_dir = save_directory
    mydir = save_folder_path

    for path in video_paths:
        image_count = save_frame(path, save_dir, gap = extraction_rate_s)
    
    ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
    
    width = frameSize[0]
    height = frameSize[1]
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = checkerboard_squares_mm
    objp = objp * size_of_chessboard_squares_mm


    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = np.zeros((image_count,height,width,3))
    i=0
    for file in os.listdir(save_folder_path):
        if file.endswith(".jpeg") or file.endswith(".jpg") or file.endswith(".JPEG"):
            filename = save_folder_path+'/'+file
            # Getting the image
            y = np.array(Image.open(filename))
            '''
            # resize the image
            width = 504
            height = 378
            dsize = (width, height)
            output = cv.resize(y, dsize)
            '''     
            images[i] = y

            i += 1
    print('------saving-rgb-images--------------------------------')
    # Saves the files to your current directory
    np.save('images', images)
    print(len(images))
    print('----------------------DONE-----------------------------')
    print('-------------------------------------------------------')

    for image in images:

        img = image.astype('uint8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
            #plt.imshow(img) # not displaying for some reason
            cv2.waitKey(1000)


    #print(image.shape)
    #plt.imshow(image.astype('uint8'))

    cv2.destroyAllWindows()    

    ############## CALIBRATION #######################################################

    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    
    ############## UNDISTORTION #####################################################

    img = plt.imshow(images[5].astype('uint8'))
    img = images[5]
    h,  w = img.shape[:2] # getting height and width to use in optimization function
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))    
    
    # Reprojection Error
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )
    projectionError = mean_error/len(objpoints)
    
    return projectionError, newCameraMatrix, dist

 




