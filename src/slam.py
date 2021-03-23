#!/usr/bin/env python3

# TODO: find ways to integrate g2o

import cv2
#import g2o
import numpy as np

from display import Display, Frame, Map, Point
from extractor import denormalise, match_frames

F = 270
W, H = 1920//2, 1080//2
K = np.array([
  [F, 0, W//2],
  [0, F, H//2],
  [0, 0, 1]
])
disp = Display(W, H)
map = Map()

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

def process_frame(img):
  """
  Draws circles on video feed to visualise extracted features.

  Parameters:
    img (numpy.ndarray): image to process
  """
  img = cv2.resize(img, (W, H))
  frame = Frame(map, img, K)

  if frame.id == 0:
    return

  f1 = map.frames[-1]
  f2 = map.frames[-2]

  idx1, idx2, Rt = match_frames(f1, f2)
  f1.pose = np.dot(Rt, f2.pose)

  # 3D coordinates
  pts4d = triangulate(
    f1.pose,
    f2.pose,
    f1.pts[idx1],
    f2.pts[idx2]
  )
    
  pts4d /= pts4d[:, 3:]

  # remove points behind camera and with little parallax
  good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

  # create 3D points
  for i,p in enumerate(pts4d):
    if not good_pts4d[i]:
      continue

    pt = Point(p)
    pt.add_observation(f1, idx1[i])
    pt.add_observation(f2, idx2[i])

  # display matches and connect them by a line
  for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
    u1, v1 = denormalise(K, pt1)
    u2, v2 = denormalise(K, pt2)
    cv2.circle(img, (u1, v1), color=(0,255,50), radius=3)
    cv2.line(img, (u1,v1), (u2,v2), color=(255,0,0))

  disp.paint(img)
  map.display()

if __name__ == "__main__":
  cap = cv2.VideoCapture("./assets/Seoul Bike Ride POV.mp4")

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      break