import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading

class PlanParser(Node):
    def __init__(self):
        super().__init__('plan_parser')

        # parameters
        self.declare_parameter('solution_file', '/data/plans/task01.pddl.soln')
        self.solution_file_path = self.get_parameter('solution_file').get_parameter_value().string_value

        # plan and tracking
        self.plan_lines = []
        self.current_index = 0
        self.waiting_for_response = threading.Event()

        # publishers and subscriber
        self.step_publisher = self.create_publisher(String, 'plan_step', 10)
        self.subscriber = self.create_subscription(String, 'action_done', self.action_done_callback, 10)
        self.done_publisher = self.create_publisher(String, 'action_done', 10)

        self.load_plan()

        # timer repeats until subscriber is found, then publishes initial step
        self.wait_timer = self.create_timer(1.0, self.wait_for_subscriber)

    def load_plan(self):
        try:
            with open(self.solution_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith(';'):
                        self.plan_lines.append(line)
            self.get_logger().info(f"Loaded {len(self.plan_lines)} steps from the plan.")
        except Exception as e:
            self.get_logger().error(f"Failed to load plan: {e}")

    def wait_for_subscriber(self):
        if self.request_publisher.get_subscription_count() == 0:
            self.get_logger().info(f"Waiting for subscriber.")
            return
        self.wait_timer.cancel()
        self.process_next_step()

    def process_next_step(self):
        if self.current_index >= len(self.plan_lines):
            self.get_logger().info("Plan execution complete.")
            return

        line = self.plan_lines[self.current_index]
        self.current_index += 1

        if line.startswith("place-loop"):
            parts = line.split()
            if len(parts) > 1:
                hook = parts[1]
                self.place_loop(hook)
        elif line == "request-loop":
            self.request_loop()

    def request_loop(self):
        self.get_logger().info("Executing request-loop action")
        self.waiting_for_response.set()
        threading.Thread(target=self.wait_for_keypress).start()

    def wait_for_keypress(self):
        input("Press Enter to continue...")  # wait for user confirmation
        msg = String()
        msg.data = f"Loop given to gripper."
        self.done_publisher.publish(msg)

    def place_loop(self, hook):
        self.get_logger().info(f"Sending place-loop instruction with hook {hook}")
        msg = String()
        msg.data = f"place-loop {hook}"
        self.step_publisher.publish(msg)

    def action_done_callback(self, msg):
        self.get_logger().info(f"Received action done: {msg.data}")
        self.waiting_for_response.clear()
        self.process_next_step()

def main(args=None):
    rclpy.init(args=args)
    node = PlanParser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
