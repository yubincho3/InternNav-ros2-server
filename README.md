# InternNav-ros2-server

GPU inference workspace for InternNav.
Runs two cooperative nodes that together perform vision-language navigation (VLN) reasoning and continuous trajectory generation for the Unitree Go2 robot.

## 🔧 Prerequisites

### TensorRT
- System1 requires TensorRT to be available on the system.

### zenoh-bridge-ros2dds
- Install and run zenoh-bridge-ros2dds on both the server and client machines.
- Refer to the [official documentation](https://zenoh.io/docs/getting-started/installation/).
  - The version must match on both the server and client.

```bash
echo '
TODO!!!
' >> <path to zenoh config>
```
>  Zenoh config file should be a '.json5' format

---

## 📦 Packages

| Package | Build Type | Description |
|---------|------------|-------------|
| `internnav_server` | ament_python | ROS 2 nodes — System1 and System2 |
| `internnav_server_interfaces` | ament_cmake | Internal message types (Latent, PlanContext) |

---

## 🤖 Nodes

### `internnav_system2` — VLN Understanding

Runs InternVLA-N1 (a multimodal VLN model built on Qwen2.5 VL) to interpret natural language navigation instructions against a rolling window of RGB observations.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_path` | string | — | Path to the System2 model directory (output of `split_model.py`) |
| `device` | string | `cuda:0` | CUDA device for inference |
| `rgb_topic` | string | — | Input camera topic name |
| `resize_w` | int | `384` | Target width to resize images before inference |
| `resize_h` | int | `384` | Target height to resize images before inference |
| `num_history` | int | `8` | Max historical frames to include in prompt |
| `instruction` | string | Move to the yellow cone. | Natural language navigation goal |

**Subscribed Topics**

| Topic | Type | Description |
|-------|------|-------------|
| `{rgb_topic}` | `sensor_msgs/Image` | Live RGB stream |
| `/internnav/server/system2/instruction` | `std_msgs/String` | Updates navigation instruction at runtime |
| `/internnav/server/cmd_reset` | `std_msgs/Empty` | Resets internal state |

**Published Topics**

| Topic | Type | Description |
|-------|------|-------------|
| `/internnav/server/system2/plan_context` | `internnav_server_interfaces/PlanContext` | Latent features + reference image forwarded to System1 |
| `/internnav/server/system2/output_discretes` | `internnav_interfaces/DiscreteStamped` | Discrete action sequence for the planner |

---

### `internnav_system1` — Trajectory Generation

Runs a TensorRT-optimized DiT-based trajectory generation model that converts the latent context from System2 into a 2D trajectory in the robot body frame.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_path` | string | — | Path to the TensorRT `.engine` file (output of `convert_sys1_trt.py`) |
| `device` | string | `cuda:0` | CUDA device for inference |
| `rgb_topic` | string | — | Input camera topic name |

**Subscribed Topics**

| Topic | Type | Description |
|-------|------|-------------|
| `{rgb_topic}` | `sensor_msgs/Image` | Live RGB stream |
| `/internnav/server/system2/plan_context` | `internnav_server_interfaces/PlanContext` | Latent + reference image from System2 |
| `/internnav/server/system2/output_discretes` | `internnav_interfaces/DiscreteStamped` | Triggers planning reset on new discrete action |
| `/internnav/server/cmd_reset` | `std_msgs/Empty` | Resets internal state |
| `/utlidar/robot_odom` | `nav_msgs/Odometry` | Robot odometry |

**Published Topics**

| Topic | Type | Description |
|-------|------|-------------|
| `/internnav/server/system1/output_path` | `nav_msgs/Path` | Trajectory in `base_footprint` frame |

---

## 📨 Internal Messages

### `Latent.msg`

Serialized tensor for passing latent feature vectors between nodes.

```
int32[] shape     # tensor shape, e.g. [1, 4, 768]
float32[] data    # flattened tensor values
```

### `PlanContext.msg`

Full planning context passed from System2 to System1.

```
Latent latent                    # extracted latent features
sensor_msgs/Image reference_rgb  # reference image at decision point
uint32 s2_step                   # System2 inference step counter
```

---
## 🛠️ Build

### 1. Installation

```bash
mkdir -p InternNav_ws/src
cd InternNav_ws/src

git clone https://github.com/yubincho3/InternNav-ros2-interfaces
git clone https://github.com/yubincho3/InternNav-ros2-server
```

### 2. Environment Setup

```bash
# 1. Create conda environment
conda create -n internnav python=3.10
conda activate internnav
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate internnav

# 2. Install dependencies
cd InternNav_ws
pip install -r src/InternNav-ros2-server/requirements.txt
```

### 3. Model Setup

The original InternVLA-N1 is distributed as a single unified model (`InternVLAN1ForCausalLM`).
Before launching the ROS nodes, it must be split into System1 and System2 components, and System1 must be converted to a TensorRT engine.

### Step 1 — Download InternVLA-N1-DualVLN from Hugging Face

```bash
hf download InternRobotics/InternVLA-N1-DualVLN --local-dir <path to internvla-n1-dualvln>
```

### Step 2 — Split the model

```bash
cd InternNav_ws/InternNav-ros2-server/InternNav
python3 scripts/split_model.py \
  --model_dir <path to internvla-n1-dualvln> \
  --output_dir <path to split> \
  --device cuda:0
```

Output:
- `<path to split>/system1/model.safetensors` — trajectory generation weights
- `<path to split>/system2/` — VLN model directory (weights + tokenizer + config)

### Step 3 — Convert System1 to TensorRT

```bash
# Note: This process is time-intensive and may take upwards of 20 minutes.
cd InternNav_ws/InternNav-ros2-server/InternNav
python3 scripts/convert_sys1_trt.py \
  --model_path <path to split>/system1/model.safetensors \
  --engine_path <path to split>/system1/model.engine
```

Exports System1 to ONNX (FP32), then builds a TensorRT BF16 engine. The intermediate `.onnx` file is deleted automatically on success.

After these two steps, use the resulting paths as launch arguments:
- `s1_model_path` → `<path to split>/system1/model.engine`
- `s2_model_path` → `<path to split>/system2`

### 4. Build packages

```bash
conda activate internnav
source /opt/ros/<distro>/setup.bash

cd InternNav_ws
colcon build --symlink-install
```

---

## 🚀 Launch

```bash
# Terminal 1. Turn on zenoh bridge
zenoh-bridge-ros2dds -c <path to zenoh config>

# Terminal 2. Launch InternNav server
source /opt/ros/<distro>/setup.bash
source InternNav_ws/install/setup.bash
ros2 launch internnav_server realworld.launch.py \
  rgb_topic:=/rgb/image/topic/name \
  s1_model_path:=<path to split>/system1/model.engine \
  s2_model_path:=<path to split>/system2
```

```bash
#### Example ####
# Terminal 1
zenoh-bridge-ros2dds -c zenoh-config.json5

# Terminal 2
source /opt/ros/humble/setup.bash
source InternNav_ws/install/setup.bash
ros2 launch internnav_server realworld.launch.py \
  rgb_topic:=/camera/color/image_raw \
  s1_model_path:=checkpoints/system1/model.engine \
  s2_model_path:=checkpoints/system2
```

**Launch arguments**

| Argument | Description |
|----------|-------------|
| `rgb_topic` | ROS topic for the RGB camera stream |
| `s1_model_path` | Path to the System1 TensorRT `.engine` file (output of `convert_sys1_trt.py`) |
| `s2_model_path` | Path to the System2 model directory (output of `split_model.py`) |

System1 runs on `cuda:0`, System2 on `cuda:1` by default (configurable in the launch file).

## 👏 Acknowledgements

This project is based on [InternNav](https://github.com/InternRobotics/InternNav) by Intern Robotics.
The original codebase has been adapted from an HTTP/multi-threaded architecture to a ROS 2 architecture for real-world deployment on the Unitree Go2.

## 📄 License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
