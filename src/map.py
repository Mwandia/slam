import numpy as np
import OpenGL.GL as gl
import pangolin
from multiprocessing import Process, Queue

class Map(object):
  
  def __init__(self):
    self.frames = []
    self.points = []
    self.state = None
    self.q = Queue()

    p = Process(target=self.viewer_thread, args=(self.q))
    p.daemon=True
    p.start()

  def display_thread(self, q):
    self.display_init()
    while True:
      self.display_refresh(q)

  def display(self):
    poses, pts = [], []
    for f in self.frames:
     poses.append(f.pose)
    
    for p in self.points:
      pts.append(p.pt)
    
    self.q.put((poses, pts))
  
  def display_init(self):

    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pangolin.OpenGlRenderState(
      pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
      pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY)
    )
    self.handler = pangolin.Handler3D(self.scam)

    # interactive view
    self.dcame = pangolin.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    self.dcam.setHandler(self.handler)

  def display_refresh(self, q):
    if self.state is None or not q.empty():
      self.state = q.get()

    ppts = np.array([d[:3, 3] for d in self.state[0]])
    spts = np.array(self.state[1])

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    self.dcam.Activate(self.scam)

    gl.glPointSize(10)
    gl.glColor3f(0.0, 1.0, 0.0)
    pangolin.DrawPoints(ppts)

    gl.glPointSize(2)
    gl.glColor3f(0.0, 1.0, 0.0)
    pangolin.DrawPoints(d for d in self.state[1])
    pangolin.DrawPoints(spts)

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
