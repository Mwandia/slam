import cv2
import numpy as np
import sdl2
import sdl2.ext

from scipy.spatial import KDTree

from extractor import extract, normalise

class Display(object):
  """
  Creates a window to view images/videos and graphics applied to the video

  Attributes:
  -----------
    H (int): height of display window
    W (int): width of display window

  Methods:
  --------
    paint(numpy.ndarray): draw point on window
  """
  def __init__(self, W: int, H: int):
    sdl2.ext.init()

    self.H, self.W = H, W
    self.window = sdl2.ext.Window('3D reconstruction', size=(W,H))

    self.surface = self.window.get_surface()
    self.window.show()

  def paint(self, img):
    """
    Parameters:
    -----------
      img (numpy.ndarray): image to draw on and display
    """
    events = sdl2.ext.get_events()
    for event in events:
      if event.type == sdl2.SDL_QUIT:
        exit(0)
    
    # draw on window
    surf = sdl2.ext.pixels2d(self.surface)
    surf[:] = img.swapaxes(0,1)[:, :, 0]

    # refresh
    self.window.refresh()

class Frame(object):
  
  def __init__(self, map, img, k):    
    self.img = img
    self.K = k
    self.Kinv = np.linalg.inv(self.K)
    self.pose = np.eye(4)
    self.h, self.w = img.shape[0:2]

    kps, self.des = extract(img)
    self.kd = KDTree(self.kps)

    if kps is not None:
      self.kps = normalise(self.Kinv, kps)
      
    self.pts = [None]*len(self.kps)

    self.id = len(map.frames)
    map.frames.append(self)

  def __bool__(self):
    gray_version = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray_version) == 0:
      return False
    return True
