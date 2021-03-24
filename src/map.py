import numpy as np
import OpenGL.GL as gl
import pangolin
from multiprocessing import Process, Queue

class Map(object):
  
  def __init__(self):
    self.frames = []
    self.points = []
    self.state = None
    self.q = None

  def create_display(self):
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
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
    self.dcam.setHandler(self.handler)

  def display_refresh(self, q):
    if self.state is None or not q.empty():
      self.state = q.get()

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    self.dcam.Activate(self.scam)

    # draw poses
    gl.glPointSize(10)
    gl.glColor3f(0.0, 1.0, 0.0)
    pangolin.DrawPoints(self.state[0])

    # draw keypoints
    gl.glPointSize(2)
    gl.glColor3f(1.0, 0.0, 0.0)
    pangolin.DrawPoints(self.state[1])

    pangolin.FinishFrame()

  def display(self):
    if self.q is None:
      return
      
    poses, pts = [], []
    for f in self.frames:
     poses.append(f.pose)
    
    for p in self.points:
      pts.append(p.pt)
    
    self.q.put((np.array(poses), np.array(pts)))

class Point(object):
  
  def __init__(self, map, loc):
    self.location = loc
    self.frames = []
    self.idxs = []
    self.id = len(map.points)
    map.points.append(self)

  def add_observation(self, frame, idx):
    frame.pts[idx] = self
    self.frames.append(frame)
    self.idxs.append(idx)
