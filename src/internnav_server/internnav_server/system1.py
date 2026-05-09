import sys
from typing import Optional

import cv2
import numpy as np
import torch

# ros2
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ros2 msgs
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

# User defined msgs
from internnav_interfaces.msg import DiscreteStamped
from internnav_server_interfaces.msg import PlanContext

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / 'InternNav'))

import internnav_server.utils as utils
from internnav.model.basemodel.internvla_n1.trt.system1_runner import TRTSystem1Runner

class System1(LifecycleNode):
    def __init__(self):
        super().__init__('internnav_system1')

        self.declare_parameter('rgb_topic', '')
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cuda:0')

        self._model: Optional[TRTSystem1Runner] = None

    @property
    def device(self):
        return self.__device

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self._rgb_topic = self.get_parameter('rgb_topic')\
            .get_parameter_value()\
            .string_value
        model_path = self.get_parameter('model_path')\
            .get_parameter_value()\
            .string_value
        self.__device = self.get_parameter('device')\
            .get_parameter_value()\
            .string_value

        self._load_model(model_path)
        self._reset_state()

        self._path_pub = self.create_lifecycle_publisher(
            Path,
            '/internnav/server/system1/output_path',
            1
        )
        self._cmd_reset_sub = self.create_subscription(
            Empty,
            '/internnav/server/cmd_reset',
            self.reset,
            1
        )

        self.get_logger().info('System1 configured')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._image_sub = self.create_subscription(
            Image,
            self._rgb_topic,
            self.image_callback,
            qos
        )
        self._plan_sub = self.create_subscription(
            PlanContext,
            '/internnav/server/system2/plan_context',
            self.plan_callback,
            1
        )
        self._discretes_sub = self.create_subscription(
            DiscreteStamped,
            '/internnav/server/system2/output_discretes',
            self.discretes_callback,
            1
        )

        self.get_logger().info('System1 activated')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.destroy_subscription(self._image_sub)
        self.destroy_subscription(self._plan_sub)
        self.destroy_subscription(self._discretes_sub)
        self._reset_state()
        self.get_logger().info('System1 deactivated')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.destroy_subscription(self._cmd_reset_sub)
        self.destroy_publisher(self._path_pub)
        self._model = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    def _load_model(self, model_path: str):
        self.get_logger().info('Loading System1 model...')
        self._model = TRTSystem1Runner(engine_path=model_path)

        self.get_logger().info('Warming up System1 model...')
        latents_in = torch.randn(1, 4, 768, device=self.device, dtype=torch.float32)
        images_in = torch.randn(1, 2, 224, 224, 3, device=self.device, dtype=torch.float32)
        noise_in = torch.randn(1, 32, 3, device=self.device, dtype=torch.float32)
        for _ in range(5):
            self._model.generate_traj(
                latents_in, images_in, noise=noise_in,
                num_inference_steps=10, num_sample_trajs=1
            )

    def _reset_state(self):
        self._latest_latent: Optional[torch.Tensor] = None
        self._latest_ref_tensor: Optional[torch.Tensor] = None
        self._last_s2_step: int = -1
        self._plan_warned = False

    def reset(self, _=None):
        self._reset_state()
        self.get_logger().info('System1 state reset')

    def discretes_callback(self, _):
        if self._last_s2_step == -1:
            return

        self.get_logger().info('Discrete action received, resetting state')
        self._reset_state()

    def plan_callback(self, msg: PlanContext):
        self._latest_latent = torch.tensor(
            msg.latent.data,
            dtype=torch.float32,
            device=self.device
        ).reshape(*msg.latent.shape)

        ref_img = utils.imgmsg_to_cv2(msg.reference_rgb, desired_encoding='rgb8')
        self._latest_ref_tensor = torch.from_numpy(ref_img)\
            .to(self.device, dtype=torch.float32) / 255

        self._last_s2_step = msg.s2_step
        self.get_logger().info(f'[Step {msg.s2_step}] New latent received')

    def image_callback(self, msg: Image):
        if self._last_s2_step == -1:
            if not self._plan_warned:
                self.get_logger().warn('Result of S2 not yet received, skipping S1 inference')
                self._plan_warned = True

            return

        raw_img = utils.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        img = cv2.resize(raw_img, (224, 224))
        img_tensor = torch.from_numpy(img)\
            .to(self.device, dtype=torch.float32) / 255

        rgbs = torch.stack([self._latest_ref_tensor, img_tensor])\
            .unsqueeze(0)\
            .to(self.device, dtype=torch.float32)

        traj_latents = self._latest_latent.to(self.device, dtype=torch.float32)
        noise_in = torch.randn(1, 32, 3, device=self.device, dtype=torch.float32)

        dp_actions_np = self._model.generate_traj(
            traj_latents=traj_latents, 
            images_dp=rgbs, 
            noise=noise_in
        ).cpu().numpy()[0]

        dp_actions_np[:, :2] /= 4.0
        cumsum_xy = np.cumsum(dp_actions_np[:, :2], axis=0)

        traj_result = np.zeros((33, 2))
        traj_result[1:] = cumsum_xy

        path_msg = Path()
        path_msg.header.frame_id = 'base_footprint'
        path_msg.header.stamp = msg.header.stamp

        for i in range(traj_result.shape[0]):
            pose = PoseStamped()

            pose.header = path_msg.header
            pose.pose.position.x = float(traj_result[i, 0])
            pose.pose.position.y = float(traj_result[i, 1])

            path_msg.poses.append(pose)

        self._path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = System1()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
