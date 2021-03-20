#!/usr/bin/env python3

# TODO: find ways to integrate g2o

import cv2
import numpy as np

from display import Display
from extractor import Frame, denormalise, match

W, H= 1920//2, 1080//2
F = 270
K = np.array([
  [F, 0, W//2],
  [0, F, H//2],
  [0, 0, 1]
])
frames = []

disp = Display(W, H)

def process_frame(img):
  """
  Draws circles on video feed to visualise extracted features.

  Parameters:
    img (numpy.ndarray): image to process
  """
  img = cv2.resize(img, (W, H))
  frame = Frame(img, K)
  frames.append(frame)

  if len(frames) <= 1:
    return


  ret, Rt = match(frames[-1], frames[-2])

  for pt1, pt2 in ret:
    u1, v1 = denormalise(K, pt1)
    u2, v2 = denormalise(K, pt2)
    cv2.circle(img, (u1, v1), color=(0,255,50), radius=3)
    cv2.line(img, (u1,v1), (u2,v2), color=(255,0,0))

  disp.paint(img)

if __name__ == "__main__":
  cap = cv2.VideoCapture("./assets/Seoul Bike Ride POV.mp4")

  while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
      process_frame(frame)
    else:
      break