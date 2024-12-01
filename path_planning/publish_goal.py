import rclpy
from sampling_based_inspection_coverage import path_finding
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints

class PublishGoal(Node):

    def __init__(self):
        super().__init__('publish_goal_to_nav2')

        # Initialize the action client for sending waypoints to the robot
        self.client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self.poses = self.generate_goals()  # Generate the goals
        self.publish_goals()  # Publish the goals to the action server

    def generate_goals(self):
        poses = []
        
        # Retrieve coordinates and directions from the path_finding function
        coordinates_result, directions_result = path_finding()

        # Generate PoseStamped messages for each coordinate and direction pair
        for coordinate, direction in zip(coordinates_result, directions_result):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = coordinate[0]
            pose.pose.position.y = coordinate[1]
            pose.pose.position.z = 0.0

            # Transform the direction values into quaternion orientation
            trans_direction = self.transform_direction(direction)
            pose.pose.orientation.x = trans_direction[0]
            pose.pose.orientation.y = trans_direction[1]
            pose.pose.orientation.z = trans_direction[2]
            pose.pose.orientation.w = trans_direction[3]
            poses.append(pose)

        return poses

    def transform_direction(self, direction):
        # Predefined mapping for converting direction to quaternion orientation
        z_w_map = {
            (1, 1): [0.0, 0.0, 0.5, 1.0],
            (1, 0): [0.0, 0.0, 0.0, 1.0],
            (1, -1): [0.0, 0.0, -0.5, 1.0],
            (0, 1): [0.0, 0.0, 0.6, 0.8],
            (0, -1): [0.0, 0.0, -0.6, 0.8],
            (-1, 1): [0.0, 0.0, 0.8, 0.5],
            (-1, 0): [0.0, 0.0, 1.0, 0.0],
            (-1, -1): [0.0, 0.0, -0.8, 0.5],
        }
        
        # Return the corresponding quaternion values or default to [0.0, 0.0, 0.0, 0.0]
        return z_w_map.get((direction[0], direction[1]), [0.0, 0.0, 0.0, 0.0])

    def publish_goals(self):
        # Wait for the action server to become available
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Action server not available after waiting')
            rclpy.shutdown()
            return

        # Prepare the FollowWaypoints goal message
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = self.poses

        self.get_logger().info('Publishing goal to robot')

        # Send the goal to the action server and set callbacks
        self.client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        ).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal was rejected by server')
            return

        self.get_logger().info('Goal accepted by server, waiting for result')
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        # Log the remaining distance to the current waypoint
        feedback = feedback_msg.feedback
        self.get_logger().info(f'current_waypoint: {feedback.current_waypoint}')

    def result_callback(self, future):
        result = future.result()
        # Handle the result status of the goal
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal succeeded')
        elif result.status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error('Goal was aborted')
        elif result.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().error('Goal was canceled')
        else:
            self.get_logger().error('Unknown result code')


def main(args=None):
    # Initialize the ROS client library and the node
    rclpy.init(args=args)
    node = PublishGoal()
    rclpy.spin(node)  # Keep the node running
    rclpy.shutdown()  # Shut down the ROS client library


if __name__ == '__main__':
    main()
