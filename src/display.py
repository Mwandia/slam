import cv2
import numpy as np
import sdl2
import sdl2.ext

from extractor import extract, normalise, IRt

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
    self.window = sdl2.ext.Window('3D reconstruction', size=(W,H), position=(-500,-500))

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
    self.pose = IRt
    pts, self.des = extract(img)

    if pts is not None:
      self.pts = normalise(self.Kinv, pts)

    self.id = len(map.frames)
    map.frames.append(self)

  def __bool__(self):
    gray_version = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray_version) == 0:
      return False
    return True

class Map(object):
  
  def __init__(self):
    self.frames = []
    self.points = []
    self.display_init()

  def display(self):
    poses, pts = [], []
    for f in self.frames:
     poses.append(f.pose)
    
    for p in self.points:
      pts.append(p.pt)
    
    self.state = poses, pts
    self.display_refresh()
  
  def display_init(self):
    import OpenGL.GL as gl
    import pangolin

    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pangolin.OpenGlRenderState(
      pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100)
      pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY)
    )
    self.handler = pangolin.Handler3D(self.scam)

    # interactive view
    self.dcame = pangolin.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    self.dcam.setHandler(self.handler)

  def display_refresh(self):
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    self.dcam.Activate(self.scam)

    gl.glPointSize(10)
    gl.glColor3f(0.0, 1.0, 0.0)
    pangolin.DrawPoints(d[:3, 3] for d in self.state[0])

    gl.glPointSize(2)
    gl.glColor3f(0.0, 1.0, 0.0)
    pangolin.DrawPoints(d for d in self.state[1])

    pangolin.FinishFrame()

class Point(object):
  
  def __init__(self, map, loc):
    self.location = loc
    self.frames = []
    self.idxs = []
    self.id = len(map.points)
    map.points.append(self)

  def add_observation(self, frame, idx):
    self.frames.append(frame)
    self.idxs.append(idx)
