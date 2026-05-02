import os
import re
import sys

from contextlib import redirect_stderr
with open(os.devnull, 'w') as f, redirect_stderr(f):
    from cv_bridge import CvBridge

import numpy as np
from PIL import Image as PILImage

import torch
from transformers import AutoProcessor

# ros2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ros2 msgs
from std_msgs.msg import Empty, String
from sensor_msgs.msg import Image

# User defined msgs
from internnav_interfaces.msg import DiscreteStamped
from internnav_server_interfaces.msg import Latent, PlanContext

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / 'InternNav'))

from internnav.model.basemodel.internvla_n1.internvla_n1_system2 import InternVLAN1System2

_ACTION_MAP = {'STOP': 0, '↑': 1, '←': 2, '→': 3, '↓': 5}
_COORD_PATTERN = re.compile(r'^(\d{1,3}) (\d{1,3})$')
_ACTION_PATTERN = re.compile(r'^(STOP|[↑←→↓]{1,4})$')

class System2(Node):
    def __init__(self):
        super().__init__('internvla_n1_system2')

        self.cv_bridge = CvBridge()

        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('resize_w', 384)
        self.declare_parameter('resize_h', 384)
        self.declare_parameter('num_history', 8)
        self.declare_parameter('rgb_topic', '')
        self.declare_parameter('instruction', 'Move to the yellow cone.')

        model_path = self.get_parameter('model_path')\
            .get_parameter_value().string_value
        self.device = self.get_parameter('device')\
            .get_parameter_value().string_value
        self.resize_w = self.get_parameter('resize_w')\
            .get_parameter_value().integer_value
        self.resize_h = self.get_parameter('resize_h')\
            .get_parameter_value().integer_value
        self.num_history = self.get_parameter('num_history')\
            .get_parameter_value().integer_value
        rgb_topic = self.get_parameter('rgb_topic')\
            .get_parameter_value().string_value
        self.instruction = self.get_parameter('instruction')\
            .get_parameter_value().string_value

        # TODO: YOLO(LOVON) Integration
        # self.declare_parameter('yolo_model', 'yolo26x.pt')
        # self.declare_parameter('yolo_conf_threshold', 0.3)
        # self.declare_parameter('yolo_object_extraction_model_path')
        # self.declare_parameter('yolo_tokenizer_path')

        # yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        # self.yolo_conf_threshold = self.get_parameter('yolo_conf_threshold').get_parameter_value().double_value
        # yolo_obj_model_path = self.get_parameter('yolo_object_extraction_model_path').get_parameter_value().string_value
        # yolo_tokenizer_path = self.get_parameter('yolo_tokenizer_path').get_parameter_value().string_value

        self.get_logger().info(f'Loading System2 model...')
        self.model = InternVLAN1System2.from_pretrained_system2(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map={'': self.device},
            attn_implementation='flash_attention_2'
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        self.processor.tokenizer.padding_side = 'left'

        # TODO: torch.compile 적용
        # 못할수도?

        self._warmup()
        self.reset()

        # TODO: YOLO(LOVON) Integration
        # YOLO 통합할 땐 모델을 먼저 TensorRT로 컴파일해서 불러오기!
        # self.get_logger().info(f'Loading YOLO model ({yolo_model})...')
        # self.yolo_model_inst = None
        # self.object_extractor = None

        self.plan_ctx_pub = self.create_publisher(
            PlanContext,
            '/internnav/server/system2/plan_context',
            1
        )
        self.discretes_pub = self.create_publisher(
            DiscreteStamped,
            '/internnav/server/system2/output_discretes',
            1
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Image,
            rgb_topic,
            self.image_callback,
            qos
        )
        self.create_subscription(
            Empty,
            '/internnav/server/cmd_reset',
            self.reset,
            1
        )
        self.create_subscription(
            String,
            '/internnav/server/system2/instruction',
            self.instruction_callback,
            1
        )

        self.get_logger().info(
            'System2 node ready'
            # f'(YOLO conf={self.yolo_conf_threshold}, resize=({self.resize_w}, {self.resize_h})'
        )

    def _build_content(self, prompt_text, images):
        content = []

        img_iter = iter(images)
        for part in re.split(r'(<image>)', prompt_text):
            if part == '<image>':
                content.append({'type': 'image', 'image': next(img_iter)})
            else:
                clean = part.replace('\n', '').strip()
                if clean:
                    content.append({'type': 'text', 'text': clean})            
        
        return content

    @torch.inference_mode()
    def _warmup(self):
        self.get_logger().info('Warming up System2 model...')

        dummy_image = PILImage.new('RGB', (self.resize_w, self.resize_h), color='black')
        base_text = (
            "You are an autonomous navigation assistant. Your task is to hello. "
            "Where should you go next to stay on track? "
            "Please output the next waypoint's coordinates in the image. "
            "Please output STOP when you have successfully completed the task."
        )
        prompt_text = base_text + ' you can see <image>.'
        conversation = [{'role': 'user', 'content': self._build_content(prompt_text, [dummy_image])}]
        output_ids, inputs, _ = self._run_inference(conversation, [dummy_image])
        self.model.generate_latents(
            output_ids,
            inputs['pixel_values'],
            inputs['image_grid_thw'],
        )

    @torch.inference_mode()
    def _run_inference(self, conversation_history, input_images):
        text = self.processor.apply_chat_template(
            conversation_history, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=input_images, return_tensors='pt'
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

        llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return output_ids, inputs, llm_output

    def instruction_callback(self, msg: String):
        self.instruction = msg.data
        self.get_logger().info(f'Instruction updated: {self.instruction}')

    def image_callback(self, rgb_msg: Image):
        cv_img = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='passthrough')
        pil_img_full = PILImage.fromarray(cv_img).convert('RGB')
        pil_img = pil_img_full.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(pil_img)
        episode_idx = len(self.rgb_list) - 1

        if episode_idx == 0:
            history_ids = []
        else:
            history_ids = np.unique(
                np.linspace(0, episode_idx - 1, self.num_history, dtype=np.int32)
            ).tolist()

        base_text = (
            f"You are an autonomous navigation assistant. Your task is to {self.instruction}. "
            "Where should you go next to stay on track? "
            "Please output the next waypoint's coordinates in the image. "
            "Please output STOP when you have successfully completed the task."
        )
        prompt_text = base_text
        if history_ids:
            placeholder = '<image>\n' * len(history_ids)
            prompt_text += f' These are your historical observations: {placeholder}.'
        prompt_text += ' you can see <image>.'

        input_images = [self.rgb_list[hid] for hid in history_ids] + [pil_img]
        conversation_history = [{'role': 'user', 'content': self._build_content(prompt_text, input_images)}]

        output_ids, inputs, llm_output = self._run_inference(conversation_history, input_images)

        # Look down -> re-infer to get pixel goal
        if '↓' in llm_output:
            conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': llm_output}]}
            )
            input_images.append(pil_img_full)
            conversation_history.append({'role': 'user', 'content': [
                {'type': 'text',  'text':  'you can see'},
                {'type': 'image', 'image': pil_img_full},
                {'type': 'text',  'text':  '.'},
            ]})
            output_ids, inputs, llm_output = self._run_inference(conversation_history, input_images)
            assert llm_output != '', 'Last llm_output should not be empty when look down!'

        llm_output = llm_output.strip().upper()

        self.get_logger().info(f'[Step {self.s2_step}] LLM: {llm_output}')
        self.s2_step += 1

        if _COORD_PATTERN.fullmatch(llm_output):
            with torch.inference_mode():
                latent = self.model.generate_latents(
                    output_ids,
                    inputs['pixel_values'],
                    inputs['image_grid_thw'],
                )

            latent_msg = Latent()
            latent_msg.shape = list(latent.shape)
            latent_msg.data = latent.cpu().float().flatten().tolist()

            ctx_msg = PlanContext()
            ctx_msg.latent = latent_msg
            ctx_msg.reference_rgb = rgb_msg
            ctx_msg.s2_step = self.s2_step

            self.plan_ctx_pub.publish(ctx_msg)

        elif _ACTION_PATTERN.fullmatch(llm_output):
            if llm_output == 'STOP':
                actions = [_ACTION_MAP['STOP']]
            else:
                if '↓' in llm_output:
                    self.get_logger().warn('Look down after re-inference, dropping')
                    return

                actions = [_ACTION_MAP[c] for c in llm_output]

            discrete_msg = DiscreteStamped()
            discrete_msg.header.frame_id = 'base_footprint'
            discrete_msg.header.stamp = rgb_msg.header.stamp
            discrete_msg.actions = actions[:1] ####################
            self.discretes_pub.publish(discrete_msg)

        else:
            self.get_logger().warn('Unrecognized output, skipping')
            return

    def reset(self, _=None):
        self.s2_step = 0
        self.rgb_list = []
        torch.cuda.empty_cache()
        self.get_logger().info('System2 initialized')

def main(args=None):
    rclpy.init(args=args)
    node = System2()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
