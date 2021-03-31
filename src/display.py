import cv2
import numpy as np
import sdl2
import sdl2.ext
import pangolin
import OpenGL.GL as gl
from multiprocessing import Process, Queue
from scipy.spatial import cKDTree

from extractor import extract, normalise

class Display2D(object):
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

class Display3D(object):
  def create_display(self):
    self.state = None
    self.q = Queue()
    self.p = Process(target=self.viewer_thread, args=(self.q))
    self.p.daemon = True
    self.p.start()

  def display_thread(self, q):
    self.display_init(1024,768)
    while True:
      self.display_refresh(q)
  
  def display_init(self, w, h):

    pangolin.CreateWindowAndBind('Main', w, h)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pangolin.OpenGlRenderState(
      pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
      pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0)
    )
    self.handler = pangolin.Handler3D(self.scam)

    # interactive view
    self.dcame = pangolin.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
    self.dcam.setHandler(self.handler)

    # prevent small window init for pangolin
    self.dcam.Resize(pangolin.Viewport(0, 0, w*2, h*2))
    self.dcam.Activate()

  def display_refresh(self, q):
    while not q.empty():
      self.state = q.get()

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    self.dcam.Activate(self.scam)

    if self.state is not None:
      if self.state[0].shape[0] >= 2:
        # draw poses
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(self.state[0])

      if self.state[0].shape[0] >= 1:
        # draw current pose as yellow
        gl.glColor4f(1.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0][-1:])

      if self.state[1].shape[0] != 0:
        # draw keypoints
        gl.glPointSize(5)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(self.state[1], self.state[2])

    pangolin.FinishFrame()

  def paint(self):
    if self.q is None:
      return
      
    poses, pts, colours = [], [], []
    for f in self.frames:
     poses.append(np.linalg.inv(f.pose))
    
    for p in self.points:
      pts.append(p.pt)
      colours.append(p.colour)
    
    self.q.put((np.array(poses), np.array(pts)))

class Frame(object):
  
  def __init__(self, map, img, K, pose=np.eye(4)):    
    self.img = img
    self.K = K
    self.pose = pose
    self.h, self.w = img.shape[0:2]

    self.kps, self.des = extract(img)
    self.pts = [None]*len(self.kps)

    self.id = map.add_frame(self)

  @property
  def Kinv(self):
    if not hasattr(self, '_Kinv'):
      self._Kinv = np.linalg.inv(self.K)

    return self._Kinv

  @property
  def kps(self):
    if not hasattr(self, '_kps'):
      self._kps = normalise(self.Kinv, self.kpus)
    
    return self._kps

  @property
  def kd(self):
    if not hasattr(self, '_kd'):
      self._kd = cKDTree(self.kps)
    
    return self._kd

  def __bool__(self):
    gray_version = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray_version) == 0:
      return False
    return True
