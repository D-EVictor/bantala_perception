#!/usr/bin/env python3

from ament_index_python.packages import get_package_share_directory
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import time
import torch


TARGET_FPS = 10
DETECTION_INTERVAL = 5
WEAPON_TIMEOUT = 1.0
ALPHA = 0.3


class SimPersonTracker(Node):

    def __init__(self):
        super().__init__('sim_person_tracker')

        self.bridge = CvBridge()

        # Models
        pkg_share = get_package_share_directory('bantala_perception')
        weapon_model_path = os.path.join(pkg_share, 'models', 'gun_knife_thesis.pt')

        # Models
        self.person_model = YOLO("yolov8n.pt")
        self.weapon_model = YOLO(weapon_model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.person_model.to(device)
        self.weapon_model.to(device)
        self.get_logger().info(f"YOLO models running on {device.upper()}")

        # State
        self.frame_count = 0
        self.smoothed_cx = None
        self.smoothed_width = None
        self.last_weapon_detections = []
        self.weapon_last_seen = 0.0
        self.last_process_time = 0.0
        self.min_interval = 1.0 / TARGET_FPS

        # Publishers
        self.image_pub = self.create_publisher(CompressedImage, '/sim_cam/yolo/annotated/compressed', 10)
        self.person_pub = self.create_publisher(String, '/sim_cam/detected_persons', 10)
        self.weapon_pub = self.create_publisher(String, '/sim_cam/detected_weapons', 10)

        # Subscriber — this replaces the RTSP capture thread entirely
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',   # <-- match your Gazebo camera plugin's topic
            self.image_callback,
            10
        )

        self.get_logger().info("Sim person tracker ready.")

    def image_callback(self, msg: Image):
        now = time.time()
        if now - self.last_process_time < self.min_interval:
            return  # throttle to TARGET_FPS, same idea as yolo_loop before
        self.last_process_time = now

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_frame(frame)

    def process_frame(self, frame):
        self.frame_count += 1
        h, w = frame.shape[:2]
        annotated = frame.copy()

        # --- person detection (same logic as before) ---
        results = self.person_model.track(
            frame, imgsz=416, classes=[0], persist=True,
            tracker="bytetrack.yaml", verbose=False
        )

        largest_area = 0
        selected = None
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            for box, track_id, conf in zip(r.boxes.xyxy, r.boxes.id, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area < 1500 or conf < 0.40:
                    continue
                if area > largest_area:
                    largest_area = area
                    selected = (track_id, x1, y1, x2, y2, conf)

        persons_list = []
        if selected:
            track_id, x1, y1, x2, y2, conf = selected
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"ID {int(track_id)} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cx_norm = ((x1 + x2) / 2.0) / w
            width_norm = (x2 - x1) / w
            if self.smoothed_cx is None:
                self.smoothed_cx, self.smoothed_width = cx_norm, width_norm
            else:
                self.smoothed_cx = ALPHA * cx_norm + (1 - ALPHA) * self.smoothed_cx
                self.smoothed_width = ALPHA * width_norm + (1 - ALPHA) * self.smoothed_width

            persons_list.append(f"{int(track_id)},{self.smoothed_cx:.4f},{self.smoothed_width:.4f},{conf:.2f}")

        # --- weapon detection (same retention logic) ---
        weapons_list = []
        current_time = time.time()
        if self.frame_count % DETECTION_INTERVAL == 0:
            weapon_results = self.weapon_model(frame, imgsz=320, verbose=False)
            detected = []
            for r in weapon_results:
                if r.boxes is None:
                    continue
                for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
                    if conf < 0.40:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    detected.append({"box": (x1, y1, x2, y2), "conf": float(conf)})
            if detected:
                self.last_weapon_detections = detected
                self.weapon_last_seen = current_time

        if current_time - self.weapon_last_seen < WEAPON_TIMEOUT:
            for wpn in self.last_weapon_detections:
                x1, y1, x2, y2 = wpn["box"]
                conf = wpn["conf"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"weapon {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                weapons_list.append(f"weapon,{cx:.4f},{cy:.4f},{conf:.2f}")

        # --- publish ---
        self.person_pub.publish(String(data=";".join(persons_list)))
        self.weapon_pub.publish(String(data=";".join(weapons_list)))
        img_msg = self.bridge.cv2_to_compressed_imgmsg(annotated)
        self.image_pub.publish(img_msg)


def main():
    rclpy.init()
    node = SimPersonTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()