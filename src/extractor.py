import cv2
import numpy as np
np.set_printoptions(suppress=True)
# from numba import jit, cuda

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform

IRt = np.eye(4)

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

  ret = IRt
  ret[:3, :3] = R
  ret[:3, 3] = t
  
  return ret

def normalise(Kinv, pts):
  return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalise(K, pt):
  ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
  ret /= ret[2]
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
    return None, None

  # extracting keypoints and descriptors
  kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]
  kps, des = _ORB.compute(img, kps)

  return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des
    

def match_frames(f1, f2):    
  # matching features
  matches = _BF.knnMatch(f1.des, f2.des, k=2)
  ret, idx1, idx2 = [], [], []

  for m,n in matches:
    if m.distance < 0.75*n.distance:
      idx1.append(m.queryIdx)
      idx2.append(m.trainIdx)

      kp1 = f1.pts[m.queryIdx]
      kp2 = f2.pts[m.trainIdx]
      ret.append((kp1,kp2))

  # filter outliers
  assert len(ret) >= 8
  ret = np.array(ret)
  idx1 = np.array(idx1)
  idx2 = np.array(idx2)

  model, inliers = ransac((ret[:, 0], ret[:, 1]),
    EssentialMatrixTransform,
    min_samples=8,
    residual_threshold=0.005,
    max_trials=250)

  # get return value and 'save' previous image points
  Rt = extractRt(model.params)
    
  return idx1[inliers], idx2[inliers], Rt

class Frame(object):
  def __init__(self, img, k):    
    self.img = img
    self.K = k
    self.Kinv = np.linalg.inv(self.K)
    self.pose = IRt
    pts, self.des = extract(img)

    if pts is not None:
      self.pts = normalise(self.Kinv, pts)

  def __bool__(self):
    gray_version = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray_version) == 0:
      return False
    return True