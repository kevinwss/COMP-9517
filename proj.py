# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 21:57:38 2017

@author: Shaoshen Wang
"""

import cv2
import numpy as np

objects = []

def get_point_pairs(kp1,kp2,matches):
    points = []
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        #**********
        if (isinstance(kp1[0],tuple)):
            (x1, y1) = kp1[img1_idx]

        else:
            (x1,y1) = kp1[img1_idx].pt
        if (isinstance(kp2[0], tuple)):
            (x2, y2) = kp2[img2_idx]

        else:
            (x2, y2) = kp2[img2_idx].pt
        points.append((x1,y1,x2,y2))
    return points


def drawAllMatches(img1, img2,pairs):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1])
   # out[:rows1, :cols1] = img1
    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2])
    #out[:rows1, :cols1] = img2
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for pair in pairs:

        # Get the matching keypoints
        x1,y1,x2,y2 = pair[0],pair[1],pair[2],pair[3]
        #************

        a = np.random.randint(0,256)
        b = np.random.randint(0,256)
        c = np.random.randint(0,256)

        cv2.circle(out, (int(np.round(x1)),int(np.round(y1))), 2, (a, b, c), 1)
        cv2.circle(out, (int(np.round(x2)+cols1),int(np.round(y2))), 2, (a, b, c), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (a, b, c), 1, lineType=cv2.LINE_AA, shift=0)  #画线，cv2.line()参考官方文档

    # Also return the image if you'd like a copy
    return out

def match_new( frame, template,tracker,objects,objects_bbox,traces):
    MIN_MATCH_COUNT = 10
    # Initiate SIFT detector
    dis_thresh = 30

    kp_t_total = []
    kp_f_total = []
    total_pairs = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for idx in range(len(objects)):

        ok, bbox = tracker[idx].update(frame)
        # Draw bounding box in frame
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 0, 255))

        # Get bounding box position in template
        lp = objects_bbox[idx][0]
        rb = objects_bbox[idx][1]
        w = rb[0] - lp[0]
        h = rb[1] - lp[1]

        box = objects[idx]
        # Ccalculate sift feature
        #sift = cv2.xfeatures2d.SIFT_create()
        sift = cv2.xfeatures2d.SURF_create()
        # find the keypoints and descriptors with SIFT
        kp_t, des_t = sift.detectAndCompute(template[lp[1]:(lp[1]+h),lp[0]:(lp[0]+w)], None)
        kp_f, des_f = sift.detectAndCompute(frame, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_t, des_f, k=2)
        # Find good features
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > 10:
            src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # Revise position
            for point in src_pts:
                point[0][0] += lp[0]
                point[0][1] += lp[1]

            M, mask = cv2.findHomography(np.float32(src_pts), np.float32(dst_pts), cv2.RANSAC, 5.0)
            pts = np.float32(box).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            frame = cv2.polylines(frame, [np.int32(dst)], True, colors[idx], 3, cv2.LINE_AA)

        else:
            print("Not enough",idx)

        traces[idx].append((int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)))
        for t in traces[idx]:
            cv2.circle(frame, (t[0], t[1]), 3, colors[idx], -1)
        # revise
        # template
        kp_t_revise = []
        for point in kp_t:
            kp_t_revise.append((point.pt[0] + lp[0], point.pt[1] + lp[1]))

        # add to total
        total_pairs += get_point_pairs(kp_t_revise,kp_f,good)

    img = drawAllMatches(template, frame, total_pairs)
    return img


def get_point(event, x, y, flags, param): # Mouse event to get points of contours
    global objects
    if event == cv2.EVENT_LBUTTONDBLCLK:
        objects.append((x,y))
        print((x,y))
    if event == cv2.EVENT_RBUTTONDOWN:
        objects.append("/") #separate points


def demo(video_name):
    # Loading
    global objects
    cap = cv2.VideoCapture(video_name)
    if cap.isOpened() == False:
        print("open video failed")
    else:  
        print("open video succeeded")

    is_first_frame = True
    ret, frame = cap.read()
    objects_box = []
    objects = []
    traces = [[],[],[]] # According to #objects
    num_of_objs = 3     # According to #objects

    # Get bounding box
    for _ in range(num_of_objs):
        r = cv2.selectROI(frame)
        objects_box.append( ( (int(r[0]),int(r[1])),(int(r[0]+int(r[2])),int(r[1]+int(r[3]))) ) )
    # Get border positions

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_point)
    n_obj = 0
    last_point = 0
    while (1):
        cv2.imshow('image', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    # format transformation
    temp = []
    new_objects = []
    for obj_p in objects:
        if obj_p!="/":
            temp.append(obj_p)
        else:
            new_objects.append(temp)
            temp = []
    objects = new_objects

    # Add trackers
    trackers = []
    for idx in range(len(objects_box)):
        bbox = (objects_box[idx][0][0], objects_box[idx][0][1], objects_box[idx][1][0] - objects_box[idx][0][0],
                objects_box[idx][1][1] - objects_box[idx][0][1])  # leftup width height
        tracker = cv2.TrackerKCF_create()
        ok, frame = cap.read()
        ok = tracker.init(frame, bbox)
        trackers.append(tracker)

    # Main Loop
    while(True):
        # get a frame
        ret, frame = cap.read()
        if is_first_frame == True:
            is_first_frame = False
            template = frame
        cv2.imwrite("template.jpg", template)
        img = match_new(frame, template,trackers,objects,objects_box,traces)
        img = cv2.resize(img, (1200, 400), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("capture", img)
        #cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':

    demo('./sample2.mp4')
    # Usage:
    # Fist select the region of interest, enter to confirm
    # Second  double click to select the contours, right click to confirm
    # "q" to start
