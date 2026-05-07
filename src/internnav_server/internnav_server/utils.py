import cv2
import numpy as np

from PIL import Image as PILImage, ImageDraw, ImageFont

from sensor_msgs.msg import Image

_SUPPORTED_ENCODINGS = {'rgb8', 'bgr8'}

_FONT = ImageFont.truetype('DejaVuSansMono.ttf', 20)
_TEXT_HEIGHT = 26
_PADDING = 10
_BOX_X, _BOX_Y = 10, 10

def imgmsg_to_cv2(msg, desired_encoding='bgr8'):
    if msg.encoding not in _SUPPORTED_ENCODINGS:
        raise ValueError(f'Unsupported source encoding: "{msg.encoding}". Only rgb8/bgr8 are supported.')
    if desired_encoding not in _SUPPORTED_ENCODINGS:
        raise ValueError(f'Unsupported desired encoding: "{desired_encoding}". Only rgb8/bgr8 are supported.')

    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

    if msg.encoding != desired_encoding:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if desired_encoding == 'bgr8' else cv2.COLOR_BGR2RGB)

    return img

def cv2_to_imgmsg(img, encoding='bgr8'):
    if encoding not in _SUPPORTED_ENCODINGS:
        raise ValueError(f'Unsupported encoding: "{encoding}". Only rgb8/bgr8 are supported.')
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f'Expected HxWx3 image, got shape {img.shape}.')

    msg = Image()
    msg.height, msg.width = img.shape[:2]
    msg.encoding = encoding
    msg.step = img.strides[0]
    msg.data = img.tobytes()
    return msg

def annotate_image(
    idx: int,
    image: PILImage,
    llm_output: str,
    pixel_goal: tuple
) -> np.ndarray:
    draw = ImageDraw.Draw(image)
    text_content = [
        f'Frame    Id  : {idx}',
        f'Actions      : {llm_output}' 
    ]

    bboxes = [draw.textbbox((0, 0), line, font=_FONT) for line in text_content]
    max_width = max(b[2] - b[0] for b in bboxes)

    draw.rectangle(
        [_BOX_X, _BOX_Y,
         _BOX_X + max_width + 2 * _PADDING,
         _BOX_Y + len(text_content) * _TEXT_HEIGHT + 2 * _PADDING],
        fill='black'
    )

    y_position = _BOX_Y + _PADDING
    for line in text_content:
        draw.text((_BOX_X + _PADDING, y_position), line, fill='white', font=_FONT)
        y_position += _TEXT_HEIGHT

    np_image = np.array(image)
    cv2.circle(np_image, (pixel_goal[1], pixel_goal[0]), 5, (0, 0, 255), -1)

    return np_image
