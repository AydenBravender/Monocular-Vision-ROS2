import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import cv2
import torch
import time
import numpy as np

class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')
        
        # Q matrix - Camera parameters - Can also be found using stereoRectify
        self.Q = np.array(([1.0, 0.0, 0.0, -160.0],
                          [0.0, 1.0, 0.0, -120.0],
                          [0.0, 0.0, 0.0, 350.0],
                          [0.0, 0.0, 1.0/90.0, 0.0]),dtype=np.float32)

        # Load a MiDas model for depth estimation
        self.model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)

        # Move model to GPU if available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()

        # Load transforms to resize and normalize the image
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform

        # Open up the video capture from a webcam
        self.cap = cv2.VideoCapture(0)
        
        self.publisher_ = self.create_publisher(PointCloud2, 'point_cloud', 10)
        
        self.timer = self.create_timer(0.1, self.publish_point_cloud)
        
    def publish_point_cloud(self):
        success, img = self.cap.read()

        start = time.time()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply input transforms
        input_batch = self.transform(img).to(self.device)

        # Prediction and resize to original resolution
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Reproject points into 3D
        points_3D = cv2.reprojectImageTo3D(depth_map, self.Q, handleMissingValues=False)

        # Get rid of points with value 0 (i.e no depth)
        mask_map = depth_map > 0.4

        # Mask colors and points. 
        output_points = points_3D[mask_map]
        output_colors = img[mask_map]

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime

        # Publish the point cloud
        point_cloud_msg = PointCloud2()
        # Fill in the header
        point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        point_cloud_msg.header.frame_id = "base_link"
        # Fill in the point cloud data
        point_cloud_msg.height = 1
        point_cloud_msg.width = len(output_points)
        point_cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud_msg.is_bigendian = False
        point_cloud_msg.point_step = 16
        point_cloud_msg.row_step = 16 * len(output_points)
        point_cloud_msg.is_dense = True
        # Flatten and concatenate the 3D points and colors
        point_cloud_data = np.hstack((output_points.reshape(-1, 3), output_colors.reshape(-1, 3)))
        # Convert to list of tuples
        point_cloud_data_list = [(pt[0], pt[1], pt[2], rgb_to_float(pt[3], pt[4], pt[5])) for pt in point_cloud_data]
        # Pack the point cloud data into the message
        point_cloud_msg.data = point_cloud_data_list
        self.publisher_.publish(point_cloud_msg)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        depth_map = (depth_map*255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Image', img)
        cv2.imshow('Depth Map', depth_map)

        if cv2.waitKey(5) & 0xFF == 27:
            self.destroy_node()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    depth_estimation_node = DepthEstimationNode()
    rclpy.spin(depth_estimation_node)
    depth_estimation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
