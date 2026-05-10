from typing import Callable, Dict, List

# ros2
import rclpy
import rclpy.task
from rclpy.client import Client
from rclpy.node import Node

# ros2 msgs
from std_msgs.msg import String
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

# lifecycle
from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import Transition, State

# ros2 srvs
from std_srvs.srv import Trigger

_NODE_SEQUENCE = [
    'internnav_controller',
    'internnav_planner',
    'internnav_system1',
    'internnav_system2',
]

class InternNavManager(Node):
    def __init__(self):
        super().__init__('internnav_manager')

        self._change_state_clients: Dict[str, Client] = {
            node: self.create_client(ChangeState, f'/{node}/change_state')
            for node in _NODE_SEQUENCE
        }
        self._get_state_clients: Dict[str, Client] = {
            node: self.create_client(GetState, f'/{node}/get_state')
            for node in _NODE_SEQUENCE
        }

        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1
        )
        self._state_pub = self.create_publisher(String, '/internnav/system_state', state_qos)
        self.create_service(Trigger, '/internnav/activate', self._activate_callback)
        self.create_service(Trigger, '/internnav/deactivate', self._deactivate_callback)

        self._busy = False
        self._current_state = 'UNCONFIGURED'
        self._publish_state('INACTIVE')

    def _publish_state(self, state: str):
        msg = String()
        msg.data = state
        self._state_pub.publish(msg)
        self.get_logger().info(f'InternNav transition [{self._current_state}] -> [{state}]')
        self._current_state = state

    def _call_change_state(
        self,
        node_name: str,
        transition_id: int,
        on_success: Callable,
        on_error: Callable
    ):
        req = ChangeState.Request()
        req.transition.id = transition_id
        f = self._change_state_clients[node_name].call_async(req)

        def _cb(f: rclpy.task.Future):
            try:
                result = f.result()
            except BaseException as e:
                self.get_logger().error(f'{node_name} transition {transition_id} raised {e}')
                on_error()
                return

            if result is None or not result.success:
                self.get_logger().error(f'{node_name} transition {transition_id} failed')
                on_error()
            else:
                on_success()

        f.add_done_callback(_cb)

    def _get_state(self, node_name: str, on_done: Callable[[int], None]):
        f = self._get_state_clients[node_name].call_async(GetState.Request())

        def _cb(f: rclpy.task.Future):
            try:
                result = f.result()
                on_done(result.current_state.id if result else State.PRIMARY_STATE_UNKNOWN)
            except BaseException:
                on_done(State.PRIMARY_STATE_UNKNOWN)

        f.add_done_callback(_cb)

    def _configure_and_activate(self, node_name: str, on_success: Callable, on_error: Callable):
        def on_state(state: int):
            if state == State.PRIMARY_STATE_ACTIVE:
                on_success()
            elif state == State.PRIMARY_STATE_UNCONFIGURED:
                self._call_change_state(node_name, Transition.TRANSITION_CONFIGURE,
                    on_success=lambda: self._call_change_state(
                        node_name,
                        Transition.TRANSITION_ACTIVATE,
                        on_success,
                        on_error
                    ),
                    on_error=on_error
                )
            else:
                self._call_change_state(node_name, Transition.TRANSITION_ACTIVATE, on_success, on_error)

        self._get_state(node_name, on_state)

    def _deactivate_node(self, node_name: str, on_success: Callable, on_error: Callable):
        def on_state(state: int):
            if state == State.PRIMARY_STATE_ACTIVE:
                self._call_change_state(node_name, Transition.TRANSITION_DEACTIVATE, on_success, on_error)
            else:
                on_success()

        self._get_state(node_name, on_state)

    def _run_sequence(
        self,
        nodes: List[str],
        step_fn: Callable,
        on_success: Callable,
        on_error: Callable
    ):
        if not nodes:
            on_success()
            return

        step_fn(
            nodes[0],
            on_success=lambda: self._run_sequence(nodes[1:], step_fn, on_success, on_error),
            on_error=on_error
        )

    def _deactivate_all(self, nodes: List[str], on_done: Callable):
        if not nodes:
            on_done()
            return

        proceed = lambda: self._deactivate_all(nodes[1:], on_done)
        self._deactivate_node(nodes[0], on_success=proceed, on_error=proceed)

    def _activate_callback(self, _, response):
        if self._busy:
            response.success = False
            response.message = 'Transition already in progress'
            return response

        for node_name in _NODE_SEQUENCE:
            if not (self._change_state_clients[node_name].service_is_ready()
                    and self._get_state_clients[node_name].service_is_ready()):
                response.success = False
                response.message = f'{node_name} not available'
                return response

        self._busy = True
        self._publish_state('ACTIVATING')

        def on_success():
            self._busy = False
            self._publish_state('ACTIVE')

        def on_error():
            def after_cleanup():
                self._busy = False
                self._publish_state('ERROR')

            self._deactivate_all(_NODE_SEQUENCE, after_cleanup)

        self._run_sequence(_NODE_SEQUENCE, self._configure_and_activate, on_success, on_error)

        response.success = True
        response.message = 'Activation started'
        return response

    def _deactivate_callback(self, _, response):
        if self._busy:
            response.success = False
            response.message = 'Transition already in progress'
            return response

        for node_name in _NODE_SEQUENCE:
            if not (self._change_state_clients[node_name].service_is_ready()
                    and self._get_state_clients[node_name].service_is_ready()):
                response.success = False
                response.message = f'{node_name} not available'
                return response

        self._busy = True
        self._publish_state('DEACTIVATING')

        def on_success():
            self._busy = False
            self._publish_state('INACTIVE')

        def on_error():
            def after_cleanup():
                self._busy = False
                self._publish_state('ERROR')

            self._deactivate_all(_NODE_SEQUENCE, after_cleanup)

        self._run_sequence(_NODE_SEQUENCE, self._deactivate_node, on_success, on_error)

        response.success = True
        response.message = 'Deactivation started'
        return response

def main(args=None):
    rclpy.init(args=args)
    node = InternNavManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
