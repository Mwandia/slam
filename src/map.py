import g2o
import numpy as np
import OpenGL.GL as gl
import pangolin
from multiprocessing import Process, Queue

from extractor import poseRt

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
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    self.dcam.Activate(self.scam)

    # draw poses
    gl.glPointSize(10)
    gl.glColor3f(0.0, 1.0, 0.0)
    pangolin.DrawPoints(self.state[0])

    # draw keypoints
    gl.glPointSize(5)
    gl.glColor3f(1.0, 0.0, 0.0)
    pangolin.DrawPoints(self.state[1], self.state[2])

    pangolin.FinishFrame()

  def display(self):
    if self.q is None:
      return
      
    poses, pts, colours = [], [], []
    for f in self.frames:
     poses.append(f.pose)
    
    for p in self.points:
      pts.append(p.pt)
      colours.append(p.colour)
    
    self.q.put((np.array(poses), np.array(pts)))
  
  def optimise(self):
    # g2o optimiser
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3)
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(solver)

    robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

    # add frames to the graph
    for f in self.frames:
      pose = f.pose
      
      sbacam = g2o.SBACam(g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3]))
      sbacam.set_cam(f.K[0][0], f.K[1][1], f.K[2][0], f.K[2][1], 1.0)

      v_se3 = g2o.VertexCam()
      v_se3.set_id(f.id)
      v_se3.set_estimate(sbacam)
      v_se3.set_fixed(f.id <= 1)
      opt.add_vertex(v_se3)

    PT_ID_OFFSET = 0x10000
    # add points to frames
    for p in self.points:
      pt = g2o.VertexSBAPointXYZ()
      pt.set_id(p.id + PT_ID_OFFSET)
      pt.set_estimate(p.pt[0:3])
      pt.set_marginalized(True)
      pt.set_fixed(False)
      opt.add_vertex(pt)

      for f in p.frames:
        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, pt)
        edge.set_vertex(1, opt.vertex(f.id))
        
        uv = f.kps[f.pts.index(p)]
        
        edge.set_measurement(uv)
        edge.set_information(np.eye(2))
        edge.set_robust_kernel(robust_kernel)
        opt.add_edge(edge)

    opt.set_verbose(True)
    opt.initialize_optimization()
    opt.optimize(50)

    # put frames back
    for f in self.frames:
      est = opt.vertex(f.id).estimate()
      R = est.rotation().matrix()
      t = est.translation()
      f.pose = poseRt(R, t)

    # put points back
    for p in self.points:
      est = opt.vertex(p.id + PT_ID_OFFSET).estimate()
      p.pt = np.array(est)

class Point(object):
  
  def __init__(self, map, loc, colour):
    self.location = loc
    self.frames = []
    self.idxs = []
    self.colour = np.copy(colour)

    self.id = len(map.points)
    map.points.append(self)

  def add_observation(self, frame, idx):
    frame.pts[idx] = self
    self.frames.append(frame)
    self.idxs.append(idx)