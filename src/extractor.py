import cv2
import numpy as np
np.set_printoptions(suppress=True)
# from numba import jit, cuda

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

_CORNERS = 3000
_QUALITY = 0.01
_MIN_DISTANCE = 3

_ORB = cv2.ORB_create()
_BF = cv2.BFMatcher(cv2.NORM_HAMMING)

def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0],1))], axis=1)

def extractRt(E):
  W = np.mat([
    [0,-1,0],
    [1,0,0],
    [0,0,1]
  ] ,dtype=float)
  U,_,Vt = np.linalg.svd(E)

  if np.linalg.det(Vt) < 0:
    Vt *= -1.0

  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)

  t = U[:, 2]

  Rt = np.concatenate([R,t.reshape(3,1)], axis=1)
  return Rt

class Frame(object):
  def __init__(self, img, k):    
    self.K=k
    self.Kinv=np.linalg.inv(self.K)
    pts, self.des = extract(img)
    self.pts = normalise(self._Kinv)

def normalise(Kinv, pts):
  return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalise(K, pt):
  ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
  return int(round(ret[0])), int(round(ret[1]))

def extract(img):
  # detecting features
  features = cv2.goodFeaturesToTrack(
    np.mean(img, axis=2).astype(np.uint8),
    maxCorners=_CORNERS,
    qualityLevel=_QUALITY,
    minDistance=_MIN_DISTANCE
  )
  
  # prevents errors on extremely low-frequency images (i.e. black/white screens)
  if features is None:
    return None

  # extracting keypoints and descriptors
  kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]
  kps, des = _ORB.compute(img, kps)

  return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des
    

def match(f1, f2):    
  # matching features
  matches = _BF.knnMatch(f1.des, f2.des, k=2)
  ret = []
  for m,n in matches:
    if m.distance < 0.75*n.distance:
      kp1 = f1.pts[m.queryIdx]
      kp2 = f2.pts[m.trainIdx]
      ret.append((kp1,kp2))

  # filter outliers
  if len(ret) > 0:
    ret = np.array(ret)
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
      EssentialMatrixTransform,
      min_samples=8,
      residual_threshold=0.005,
      max_trials=250)

    ret = ret[inliers]

  # get return value and 'save' previous image points
  Rt = extractRt(model.params)
    
  return ret, Rt
