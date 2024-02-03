import cv2
import torch
import time
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import pcl
from pcl import PointCloud

# Q matrix - Camera parameters - Can also be found using stereoRectify
Q = np.array(([1.0, 0.0, 0.0, -160.0],
              [0.0, 1.0, 0.0, -120.0],
              [0.0, 0.0, 0.0, 350.0],
              [0.0, 0.0, 1.0/90.0, 0.0]),dtype=np.float32)

# Load a MiDas model for depth estimation
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Initialize ROS node
rospy.init_node('point_cloud_publisher')

# Create ROS publisher
pub = rospy.Publisher('point_cloud', PointCloud2, queue_size=10)

# Open up the video capture from a webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()

    start = time.time()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply input transforms
    input_batch = transform(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=False)

    # Get rid of points with value 0 (i.e no depth)
    mask_map = depth_map > 0.4

    # Mask colors and points
    output_points = points_3D[mask_map]
    output_colors = img[mask_map]

    # Convert point cloud data to ROS message format
    pcl_cloud = pcl.PointCloud()
    pcl_cloud.from_array(output_points.astype(np.float32))
    ros_msg = pcl_cloud.to_msg()

    # Publish point cloud data
    pub.publish(ros_msg)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
