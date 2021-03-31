#!/usr/bin/env python3

# TODO: find ways to integrate g2o

from math import e
import cv2
import numpy as np
import os
import time

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

def hamming_distance(a, b):
  r = (1 << np.arange(8))[:, None]
  return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)

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
  start_time = time.time()
  img = cv2.resize(img, (W, H))
  frame = Frame(map, img, K)

  if frame.id == 0:
    return

  f1 = map.frames[-1]
  f2 = map.frames[-2]

  idx1, idx2, Rt = match_frames(f1, f2)
  
  if frame.id < 5:
    f1.pose = np.dot(Rt, f2.pose)
  else:
    velocity = np.dot(f2.pose, np.linalg.inv(map.frames[-3].pose))
    f1.pose = np.dot(velocity, f2.pose)

  for i, idx in enumerate(idx2):
    if f2.pts[idx] is not None:
      f2.pts[idx].add_observation(f1, idx[i])

  # optimise pose
  pose_opt = map.optimise(local_window=1, fix_points=True)
  print(f"Pose: {pose_opt}")

  # search projection
  sbp_pts_count = 0
  if len(map.points) > 0:
    map_points = np.array([p.homogenous() for p in map.points])
    proj = np.dot(np.dot(K, f1.pose[:3]) , map_points.T).T
    proj = proj[:, 0:2] / proj[:, 2:]

    good_pts = (proj[:, 0] > 0) & (proj[:, 0] < W) & (proj[:, 1] > 0) & (proj[:, 1] < W)

    for i, p in enumerate(map.points):
      if not good_pts[i]:
        continue
      q = f1.kd.query_ball_point(proj[i], 5)
      for m_idx in q:
        if f1.pts[m_idx] is None:
          o_dist = hamming_distance(p.orb(), f1.des[m_idx])
          if o_dist < 32.0:
            p.add_observation(f1, m_idx)
            sbp_pts_count += 1

  # 3D coordinates
  good_pts4d = np.array([f1.pts[i] is None for i in idx1])

  # remove points with little parallax and points behind camera  
  pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
  good_pts4d &= np.abs(pts4d[:, 3]) > 0.005

  pts4d /= pts4d[:, 3:]

  pts_tri_local = np.dot(f1.pose, pts4d.T).T
  good_pts4d &= pts_tri_local[:, 2] > 0

  pts4d = np.dot(np.linalg.inv(f1.pose), pts4d.T).T

  # create 3D points
  for i,p in enumerate(pts4d):
    if not good_pts4d[i]:
      continue

    u,v = int(round(f1.kps[idx1[i], 0])), int(round(f1.kps[idx1[i], 1]))
    pt = Point(map, p[0:3], img[v, u])
    pt.add_observation(f1, idx1[i])
    pt.add_observation(f2, idx2[i])

  # display matches and connect them by a line
  for i1, i2 in zip(idx1, idx2):
    pt1 = f1.kps[i1]
    pt2 = f2.kps[i2]
    u1, v1 = denormalise(K, pt1)
    u2, v2 = denormalise(K, pt2)

    if f1.pts[i1] is not None:
      if len(f1.pts[i1].frames) >= 5:
        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
      else:
        cv2.circle(img, (u1, v1), color=(0,128,0), radius=3)
    else:
      cv2.circle(img, (u1, v1), color=(0,0,0), radius=3)
      
    cv2.line(img, (u1,v1), (u2,v2), color=(255,0,0))

  if disp is not None:
    disp.paint(img)

  if frame.id >= 4 and frame.id%5 == 0:
    err = map.optimise()
    print(f"Optimise: {err} units of error")
  
  # 3D display
  map.display()

  print(f"Map: {len(map.points)} points, {len(map.frames)} frames")
  print(f"Time: {(time.time()-start_time)*1000.0}ms")

if __name__ == "__main__":
  cap = cv2.VideoCapture("./assets/Seoul Bike Ride POV.mp4")
  map.create_viewer()

  # camera setup
  W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  F = int(os.getenv("F", "525"))
  K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
  Kinv = np.linalg.inv(K)

  if W > 1024:
    downscale = 1024.0/W
    F *= downscale
    H = int(H * downscale)
    W = 1024

  disp = Display(W, H)

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      break
