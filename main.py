# import cv2
# import numpy as np
#
# net = cv2.dnn.readNet('yolov5s.onnx')
#
#
# def format_yolov5(source):
#     # put the image in square big enough
#     col, row, _ = source.shape
#     _max = max(col, row)
#     resized = np.zeros((_max, _max, 3), np.uint8)
#     resized[0:col, 0:row] = source
#
#     # resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
#     result = cv2.dnn.blobFromImage(resized, 1 / 255.0, (640, 640), swapRB=True)
#
#     return result
#
# predictions = net.forward()
# output = predictions[0]
#
#
# def unwrap_detection(input_image, output_data):
#     class_ids = []
#     confidences = []
#     boxes = []
#
#     rows = output_data.shape[0]
#
#     image_width, image_height, _ = input_image.shape
#
#     x_factor = image_width / 640
#     y_factor =  image_height / 640
#
#     for r in range(rows):
#         row = output_data[r]
#         confidence = row[4]
#         if confidence >= 0.4:
#
#             classes_scores = row[5:]
#             _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
#             class_id = max_indx[1]
#             if (classes_scores[class_id] > .25):
#
#                 confidences.append(confidence)
#
#                 class_ids.append(class_id)
#
#                 x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
#                 left = int((x - 0.5 * w) * x_factor)
#                 top = int((y - 0.5 * h) * y_factor)
#                 width = int(w * x_factor)
#                 height = int(h * y_factor)
#                 box = np.array([left, top, width, height])
#                 boxes.append(box)
#
#     return class_ids, confidences, boxes
#
#
# indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
#
# result_class_ids = []
# result_confidences = []
# result_boxes = []
#
# for i in indexes:
#     result_confidences.append(confidences[i])
#     result_class_ids.append(class_ids[i])
#     result_boxes.append(boxes[i])
#
#
# class_list = []
# with open("classes.txt", "r") as f:
#     class_list = [cname.strip() for cname in f.readlines()]
#
# colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
#
# for i in range(len(result_class_ids)):
#
#     box = result_boxes[i]
#     class_id = result_class_ids[i]
#
#     color = colors[class_id % len(colors)]
#
#     conf  = result_confidences[i]
#
#     cv2.rectangle(image, box, color, 2)
#     cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
#     cv2.putText(image, class_list[class_id], (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
#
#

import numpy as np
import cv2
# from tqdm import tqdm

# imgL = cv2.imread('resources/left.jpg')
# imgR = cv2.imread('resources/right.jpg')
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(imgL, imgR)

i_ext = ".jpg"
chessboard_size = (6, 8)

pathL = "./resources/stereoL/"
pathR = "./resources/stereoR/"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

img_ptsL = []
img_ptsR = []
obj_pts = []

for i in range(1, 7):
    imgL = cv2.imread(pathL + "%d"%i + i_ext)
    imgL = cv2.resize(imgL, (800, 800))
    imgR = cv2.imread(pathR + "%d"%i + i_ext)
    imgR = cv2.resize(imgR, (800, 800))
    imgL_gray = cv2.imread(pathL + "%d"%i + i_ext, 0)
    imgL_gray = cv2.resize(imgL_gray, (800, 800))
    imgR_gray = cv2.imread(pathR + "%d"%i + i_ext, 0)
    imgR_gray = cv2.resize(imgR_gray, (800, 800))

    outputL = imgL.copy()
    outputR = imgR.copy()

    retR, cornersR = cv2.findChessboardCorners(outputR, chessboard_size, None)
    retL, cornersL = cv2.findChessboardCorners(outputL, chessboard_size, None)

    # print(str(retL) + ":" + str(retR))

    if retR and retL:
        obj_pts.append(objp)
        cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(outputR, chessboard_size, cornersR, retR)
        cv2.drawChessboardCorners(outputL, chessboard_size, cornersL, retL)
        cv2.imshow('cornersR', outputR)
        cv2.imshow('cornersL', outputL)
        cv2.waitKey(100)

        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)

retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, imgL_gray.shape[::1], None, None)
hL, wL = imgL_gray.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))


retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR, imgR_gray.shape[::1], None, None)
hR, wR = imgR_gray.shape[:2]
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC  # for same cameras
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR, new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::1], criteria_stereo, flags)

rectify_scale = 1
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::1], Rot, Trns, rectify_scale, (0, 0))
# cv2.stereoRectify(new_mtxL, distL, new_mtxR, distR, imgL_gray.shape[::1], Rot, Trns, rectify_scale, (0, 0))
Left_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxL, distL, rect_l, proj_mat_l, imgL_gray.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(new_mtxR, distR, rect_r, proj_mat_r, imgR_gray.shape[::-1], cv2.CV_16SC2)

cv_file = cv2.FileStorage("imroved_params2.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.release()

personL = cv2.imread("resources/personL" + i_ext, cv2.IMREAD_GRAYSCALE)
personL = cv2.resize(personL, (800, 800))
personR = cv2.imread("resources/personR" + i_ext, cv2.IMREAD_GRAYSCALE)
personR = cv2.resize(personR, (800, 800))

minDisparity = 0
numDisparities = 64
blockSize = 3
disp12MaxDiff = -1
uniquenessRatio = 5
speckleWindowSize = 0
speckleRange = 8

stereo = cv2.StereoSGBM_create(minDisparity=minDisparity, numDisparities=numDisparities, blockSize=blockSize, disp12MaxDiff=disp12MaxDiff, uniquenessRatio=uniquenessRatio, speckleWindowSize=speckleWindowSize, speckleRange=speckleRange)

disp = stereo.compute(personL, personR).astype(np.float32)
disp = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)

cv2.imshow('disparity', disp)
cv2.waitKey(0)

# step 1 - load the model

net = cv2.dnn.readNet('yolov5s.onnx')


# step 2 - feed a 640x640 image to get predictions

def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


image = cv2.imread('resources/personR' + i_ext)
image = cv2.resize(image, (800, 800))
input_image = format_yolov5(image)  # making the image square
blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (640, 640), swapRB=True)
net.setInput(blob)
predictions = net.forward()

# step 3 - unwrap the predictions to get the object detections

class_ids = []
confidences = []
boxes = []

output_data = predictions[0]

image_width, image_height, _ = input_image.shape
x_factor = image_width / 640
y_factor = image_height / 640

for r in range(25200):
    row = output_data[r]
    confidence = row[4]
    if confidence >= 0.4:

        classes_scores = row[5:]
        _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
        class_id = max_indx[1]
        if (classes_scores[class_id] > .25):
            confidences.append(confidence)

            class_ids.append(class_id)

            x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
            left = int((x - 0.5 * w) * x_factor)
            top = int((y - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            box = np.array([left, top, width, height])
            boxes.append(box)

# class_list = []
# with open("config_files/classes.txt", "r") as f:
#     class_list = [cname.strip() for cname in f.readlines()]

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

result_class_ids = []
result_confidences = []
result_boxes = []

for i in indexes:
    # print(i)
    result_confidences.append(confidences[i])
    result_class_ids.append(class_ids[i])
    result_boxes.append(boxes[i])

for i in range(len(result_class_ids)):
    # print(i)
    box = result_boxes[i]
    class_id = result_class_ids[i]
    if class_id != 0:  # not a person
        # print('not a person')
        continue

    cv2.rectangle(image, box, (0, 255, 255), 2)
    # cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1]), (0, 255, 255), -1)
    # cv2.putText(image, class_list[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
    # to do point position
    # pt = ...
    # pt = (1, 1)
    # print(box[0])
    # print(box[1])
    # print(box[2])
    # print(box[3])
    pt = (int((box[0] + box[0] + box[2]) / 2), int((box[1] + box[1] + box[3]) / 2))
    # print(pt)
    break

# cv2.imwrite("misc/kids_detection.png", image)
cv2.imshow("output", image)
cv2.waitKey()

base_offset = 297  # mm - distance between cameras
# f = new_mtxL[0][0] / 800 * 10
f = 5.23  # mm
depth = base_offset * f / disp[pt[0]][pt[1]]
print('distance to person == ' + str(depth) + ' mm')
