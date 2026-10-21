"""Minimal ROS 2 node: rectified stereo pair -> Fast-FoundationStereo -> colored PointCloud2.

Image callbacks only keep the latest synchronized left/right pair; the main thread runs
inference on whatever pair is newest, so frames arriving faster than the model are dropped.
Intrinsics and baseline are read once from the left/right camera_info topics.

  pixi run -e gsplat-policy node --ros-args -p scale:=0.5
"""
import os, sys, threading, time, array
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
import numpy as np
import torch
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import core.foundation_stereo  # registers the foundation_stereo_ori aliases the pickle refers to
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE


ZED = '/zed_left_arm/zed_node'


class FFSNode(Node):
  def __init__(self):
    super().__init__('ffs_node')
    p = lambda name, default: self.declare_parameter(name, default).value
    self.left_topic = p('left_topic', f'{ZED}/left/color/rect/image')
    self.right_topic = p('right_topic', f'{ZED}/right/color/rect/image')
    left_info_topic = p('left_info_topic', f'{ZED}/left/color/rect/image/camera_info')
    right_info_topic = p('right_info_topic', f'{ZED}/right/color/rect/image/camera_info')
    cloud_topic = p('cloud_topic', '/ffs/points')
    model_path = p('model', f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth')
    self.scale = p('scale', 1.0)                         # resize input images before inference
    self.valid_iters = p('valid_iters', 8)
    max_disp = p('max_disp', 192)
    self.zfar = p('zfar', 5.0)                           # meters
    self.zmin = p('zmin', 0.1)
    self.remove_invisible = p('remove_invisible', True)  # drop left-border pixels the right camera cannot see
    self.stride = p('stride', 1)                         # publish every n-th pixel per axis

    torch.autograd.set_grad_enabled(False)
    self.get_logger().info(f'Loading {model_path}')
    self.model = torch.load(model_path, map_location='cpu', weights_only=False)
    self.model.args.valid_iters = self.valid_iters
    self.model.args.max_disp = max_disp
    self.model.cuda().eval()

    self.infos = {}
    self.info_subs = [
      self.create_subscription(CameraInfo, left_info_topic, lambda m: self.infos.setdefault('left', m), 1),
      self.create_subscription(CameraInfo, right_info_topic, lambda m: self.infos.setdefault('right', m), 1),
    ]

    qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=2)
    subs = [message_filters.Subscriber(self, Image, t, qos_profile=qos) for t in (self.left_topic, self.right_topic)]
    self.sync = message_filters.ApproximateTimeSynchronizer(subs, queue_size=4, slop=0.005)
    self.sync.registerCallback(self.on_pair)
    self.pair = None
    self.pair_lock = threading.Lock()
    self.new_pair = threading.Event()

    self.pub = self.create_publisher(PointCloud2, cloud_topic, 1)
    self.get_logger().info(f'Waiting for camera_info on {left_info_topic} and {right_info_topic}')


  def on_pair(self, left:Image, right:Image):
    with self.pair_lock:
      self.pair = (left, right)
    self.new_pair.set()


  def setup_intrinsics(self):
    """Rectified intrinsics from the left P, baseline from the right P (P[3] = -fx * B)."""
    P_left = np.array(self.infos['left'].p).reshape(3,4)
    P_right = np.array(self.infos['right'].p).reshape(3,4)
    self.fx, self.fy, self.cx, self.cy = (P_left[0,0], P_left[1,1], P_left[0,2], P_left[1,2])
    self.baseline = -P_right[0,3] / P_right[0,0]
    for sub in self.info_subs:
      self.destroy_subscription(sub)
    self.get_logger().info(f'fx={self.fx:.2f} fy={self.fy:.2f} cx={self.cx:.2f} cy={self.cy:.2f} baseline={self.baseline:.4f} m')
    self.fx, self.fy, self.cx, self.cy = [v*self.scale for v in (self.fx, self.fy, self.cx, self.cy)]


  def to_tensor(self, msg:Image):
    """sensor_msgs/Image (rgb8/bgr8/rgba8/bgra8) -> (1,3,H,W) float RGB on the GPU, resized by `scale`."""
    C = 4 if msg.encoding in ('rgba8', 'bgra8') else 3
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width*C].reshape(msg.height, msg.width, C)
    img = torch.from_numpy(img).cuda()
    img = img[..., [2,1,0]] if msg.encoding.startswith('bgr') else img[..., :3]
    img = img.permute(2,0,1)[None].float()
    if self.scale != 1:
      img = F.interpolate(img, scale_factor=self.scale, mode='bilinear', align_corners=False)
    return img


  def infer(self, left:Image, right:Image):
    img0 = self.to_tensor(left)
    img1 = self.to_tensor(right)
    H, W = img0.shape[-2:]
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0_pad, img1_pad = padder.pad(img0, img1)
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
      disp = self.model.forward(img0_pad, img1_pad, iters=self.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
    disp = padder.unpad(disp.float()).reshape(H, W).clamp(min=0)

    vs, us = torch.meshgrid(torch.arange(H, device='cuda'), torch.arange(W, device='cuda'), indexing='ij')
    valid = disp > 0
    if self.remove_invisible:
      valid &= (us - disp) >= 0
    z = self.fx * self.baseline / disp.clamp(min=1e-6)
    valid &= (z >= self.zmin) & (z <= self.zfar)
    x = (us - self.cx) * z / self.fx
    y = (vs - self.cy) * z / self.fy

    rgb = img0[0].round().clamp(0, 255).to(torch.int32)
    rgb = ((rgb[0] << 16) | (rgb[1] << 8) | rgb[2]).view(torch.float32)
    points = torch.stack([x, y, z, rgb], dim=-1)[::self.stride, ::self.stride]
    points = points[valid[::self.stride, ::self.stride]]
    return self.make_cloud(left.header, points.cpu().numpy())


  def make_cloud(self, header, points:np.ndarray):
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points)
    msg.fields = [PointField(name=n, offset=4*i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(('x', 'y', 'z', 'rgb'))]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * len(points)
    msg.is_dense = True
    msg.data = array.array('B', points.astype(np.float32).tobytes())  # array.array skips the per-byte setter checks
    return msg


  def run(self):
    while rclpy.ok() and len(self.infos) < 2:
      time.sleep(0.1)
    self.setup_intrinsics()
    self.get_logger().info(f'Waiting for images on {self.left_topic} and {self.right_topic}. '
                           'The first inference compiles kernels and can take a while.')
    n, t_log, busy = 0, time.perf_counter(), 0.0
    while rclpy.ok():
      if not self.new_pair.wait(timeout=1.0):
        continue
      with self.pair_lock:
        left, right = self.pair
        self.new_pair.clear()
      t0 = time.perf_counter()
      cloud = self.infer(left, right)
      self.pub.publish(cloud)
      busy += time.perf_counter() - t0
      n += 1
      if time.perf_counter() - t_log > 2.0:
        dt = time.perf_counter() - t_log
        self.get_logger().info(f'{n/dt:.1f} Hz, {1000*busy/n:.1f} ms/frame, {cloud.width} points')
        n, t_log, busy = 0, time.perf_counter(), 0.0


def main():
  rclpy.init()
  node = FFSNode()
  threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
  try:
    node.run()
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
  main()
