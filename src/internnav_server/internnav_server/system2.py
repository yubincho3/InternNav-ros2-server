import re
import sys

from PIL import Image as PILImage

import numpy as np
import torch
from transformers import AutoProcessor

# ros2
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ros2 msgs
from std_msgs.msg import Empty, String
from sensor_msgs.msg import Image

# User defined msgs
from internnav_interfaces.msg import DiscreteStamped
from internnav_server_interfaces.msg import Latent, PlanContext

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3] / 'InternNav'))

import internnav_server.utils as utils
from internnav.model.basemodel.internvla_n1.internvla_n1_system2 import InternVLAN1System2

_ACTION_MAP = {'STOP': 0, '↑': 1, '←': 2, '→': 3, '↓': 5}
_COORD_PATTERN = re.compile(r'^(\d{1,3}) (\d{1,3})$')
_ACTION_PATTERN = re.compile(r'^(STOP|[↑←→↓]{1,4})$')

class System2(LifecycleNode):
    def __init__(self):
        super().__init__('internnav_system2')

        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('resize_w', 384)
        self.declare_parameter('resize_h', 384)
        self.declare_parameter('num_history', 8)
        self.declare_parameter('rgb_topic', '')
        self.declare_parameter('instruction', 'Move to the yellow cone.')

        self._model = None
        self._processor = None

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        model_path = self.get_parameter('model_path')\
            .get_parameter_value().string_value
        self._device = self.get_parameter('device')\
            .get_parameter_value().string_value
        self._resize_w = self.get_parameter('resize_w')\
            .get_parameter_value().integer_value
        self._resize_h = self.get_parameter('resize_h')\
            .get_parameter_value().integer_value
        self._num_history = self.get_parameter('num_history')\
            .get_parameter_value().integer_value
        self._rgb_topic = self.get_parameter('rgb_topic')\
            .get_parameter_value().string_value
        self._instruction = self.get_parameter('instruction')\
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

        self.get_logger().info('Loading System2 model...')
        self._model = InternVLAN1System2.from_pretrained_system2(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map={'': self._device},
            attn_implementation='flash_attention_2'
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        self._processor.tokenizer.padding_side = 'left'

        # TODO: torch.compile 적용
        # 못할수도?

        self._warmup()
        self._reset_state()

        self._plan_ctx_pub = self.create_lifecycle_publisher(
            PlanContext,
            '/internnav/server/system2/plan_context',
            1
        )
        self._discretes_pub = self.create_lifecycle_publisher(
            DiscreteStamped,
            '/internnav/server/system2/output_discretes',
            1
        )
        self._viz_pub = self.create_lifecycle_publisher(
            Image,
            '/internnav/server/debug_image',
            1
        )
        self._cmd_reset_sub = self.create_subscription(
            Empty,
            '/internnav/server/cmd_reset',
            self._reset_callback,
            1
        )
        self._instruction_sub = self.create_subscription(
            String,
            '/internnav/server/system2/instruction',
            self._instruction_callback,
            1
        )

        self.get_logger().info(
            'System2 node ready'
            # f'(YOLO conf={self.yolo_conf_threshold}, resize=({self._resize_w}, {self._resize_h})'
        )
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
            self._image_callback,
            qos
        )

        self.get_logger().info('System2 activated')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.destroy_subscription(self._image_sub)
        self._reset_state()
        self.get_logger().info('System2 deactivated')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.destroy_subscription(self._cmd_reset_sub)
        self.destroy_subscription(self._instruction_sub)
        self.destroy_publisher(self._plan_ctx_pub)
        self.destroy_publisher(self._discretes_pub)
        self.destroy_publisher(self._viz_pub)
        self._model = None
        self._processor = None
        torch.cuda.empty_cache()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

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

        dummy_image = PILImage.new('RGB', (self._resize_w, self._resize_h), color='black')
        base_text = (
            "You are an autonomous navigation assistant. Your task is to hello. "
            "Where should you go next to stay on track? "
            "Please output the next waypoint's coordinates in the image. "
            "Please output STOP when you have successfully completed the task."
        )
        prompt_text = base_text + ' you can see <image>.'
        conversation = [{'role': 'user', 'content': self._build_content(prompt_text, [dummy_image])}]
        output_ids, inputs, _ = self._run_inference(conversation, [dummy_image])
        self._model.generate_latents(
            output_ids,
            inputs['pixel_values'],
            inputs['image_grid_thw'],
        )

    @torch.inference_mode()
    def _run_inference(self, conversation_history, input_images):
        text = self._processor.apply_chat_template(
            conversation_history, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=input_images, return_tensors='pt'
        ).to(self._model.device)

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

        llm_output = self._processor.tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return output_ids, inputs, llm_output

    def _reset_state(self):
        self._s2_step = 0
        self._rgb_list = []
        if self._model is not None:
            torch.cuda.empty_cache()

    def _reset_callback(self, _=None):
        self._reset_state()
        self.get_logger().info('System2 state reset')

    def _instruction_callback(self, msg: String):
        self._instruction = msg.data
        self.get_logger().info(f'Instruction updated: {self._instruction}')

    def _image_callback(self, rgb_msg: Image):
        cv_img = utils.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        pil_img_full = PILImage.fromarray(cv_img)
        pil_img = pil_img_full.resize((self._resize_w, self._resize_h))
        self._rgb_list.append(pil_img)
        episode_idx = len(self._rgb_list) - 1

        if self._num_history >= episode_idx:
            history_ids = [*range(episode_idx)]
        elif self._num_history == 1:
            history_ids = [episode_idx - 1]
        else:
            end = episode_idx - 1
            denom = self._num_history - 1
            history_ids = [(end * i) // denom for i in range(self._num_history)]

        base_text = (
            f"You are an autonomous navigation assistant. Your task is to {self._instruction}. "
            "Where should you go next to stay on track? "
            "Please output the next waypoint's coordinates in the image. "
            "Please output STOP when you have successfully completed the task."
        )
        prompt_text = base_text
        if history_ids:
            placeholder = '<image>\n' * len(history_ids)
            prompt_text += f' These are your historical observations: {placeholder}.'
        prompt_text += ' you can see <image>.'

        input_images = [self._rgb_list[hid] for hid in history_ids] + [pil_img]
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

        self.get_logger().info(f'[Step {self._s2_step}] LLM: {llm_output}')

        if _COORD_PATTERN.fullmatch(llm_output):
            with torch.inference_mode():
                latent = self._model.generate_latents(
                    output_ids,
                    inputs['pixel_values'],
                    inputs['image_grid_thw'],
                )

            latent_msg = Latent()
            latent_msg.shape = list(latent.shape)
            latent_msg.data = latent.cpu().float().flatten().tolist()

            ref_img = utils.cv2_to_imgmsg(
                np.array(pil_img_full.resize((224, 224))),
                encoding='rgb8'
            )
            ref_img.header = rgb_msg.header

            ctx_msg = PlanContext()
            ctx_msg.latent = latent_msg
            ctx_msg.reference_rgb = ref_img
            ctx_msg.s2_step = self._s2_step

            self._plan_ctx_pub.publish(ctx_msg)

            if self._viz_pub.get_subscription_count() > 0:
                viz_msg = utils.cv2_to_imgmsg(
                    utils.annotate_image(
                        episode_idx,
                        pil_img_full,
                        llm_output,
                        pixel_goal=tuple(map(int, llm_output.split()))
                    ),
                    encoding='rgb8'
                )
                viz_msg.header = rgb_msg.header
                self._viz_pub.publish(viz_msg)

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
            self._discretes_pub.publish(discrete_msg)

        else:
            self.get_logger().warn('Unrecognized output, skipping')
            return

        self._s2_step += 1

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
