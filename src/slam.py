#!/usr/bin/env python3

# TODO: find ways to integrate g2o

import cv2
import numpy as np
import os
import sys

from display import Display, Frame
from extractor import denormalise, match_frames
from map import Map, Point

F = int(os.getenv("F", "800"))
W, H = 1920//2, 1080//2
K = np.array([
  [F, 0, W//2],
  [0, F, H//2],
  [0, 0, 1]
])

map = Map()
disp = None
if os.getenv("D2D") is not None:
  disp = Display(W, H) 


def triangulate(pose1, pose2, pts1, pts2):
  ret = np.zeros((pts1.shape[0], 4))
  pose1 = np.linalg.inv(pose1)
  pose2 = np.linalg.inv(pose2)

  for i, p in enumerate(zip(pts1,pts2)):
    A = np.zeros((4,4))
    A[0] = p[0][0] * pose1[2] - pose1[0]
    A[1] = p[0][0] * pose1[2] - pose1[0]
    A[2] = p[0][0] * pose1[2] - pose1[0]
    A[3] = p[0][0] * pose1[2] - pose1[0]
    _,_, vt = np.linalg.svd(A)
    ret[i] = vt[3]

  return ret

def process_frame(img):

  img = cv2.resize(img, (W, H))
  frame = Frame(map, img, K)

  if frame.id == 0:
    return

  f1 = map.frames[-1]
  f2 = map.frames[-2]

  idx1, idx2, Rt = match_frames(f1, f2)
  f1.pose = np.dot(Rt, f2.pose)

  for i, idx in enumerate(idx2):
    if f2.pts[idx] is not None:
      f2.pts[idx].add_observation(f1, idx[i])

  # 3D coordinates
  pts4d = triangulate(
    f1.pose,
    f2.pose,
    f1.kps[idx1],
    f2.kps[idx2]
  )
    
  pts4d /= pts4d[:, 3:]

  # remove points behind camera and with little parallax
  unmatched_pts = np.array([f1.pts[i] is None for i in idx1])
  good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_pts

  # create 3D points
  for i,p in enumerate(pts4d):
    if not good_pts4d[i]:
      continue

    u,v = int(round(f1.kps[idx1[i], 0])), int(round(f1.kps[idx1[i], 1]))
    pt = Point(map, p, img[v, u])
    pt.add_observation(f1, idx1[i])
    pt.add_observation(f2, idx2[i])

  # display matches and connect them by a line
  for pt1, pt2 in zip(f1.kps[idx1], f2.kps[idx2]):
    u1, v1 = denormalise(K, pt1)
    u2, v2 = denormalise(K, pt2)
    cv2.circle(img, (u1, v1), color=(0,255,50), radius=3)
    cv2.line(img, (u1,v1), (u2,v2), color=(255,0,0))

  if disp is not None:
    disp.paint(img)

  if frame.id >= 4:
    map.optimise()
  
  # 3D display
  map.display()

if __name__ == "__main__":
  cap = cv2.VideoCapture("./assets/Seoul Bike Ride POV.mp4")

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      break
