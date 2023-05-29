import cv2
import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from openvino.inference_engine import IECore

class ImageDisplay(Node):
    def __init__(self):
        super().__init__('image_display')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/sensor_fusion_pkg/overlay_msg',
            self.image_callback,
            10)
        self.subscription

        # IR 모델 로드
        model_xml = '/root/deepracer_ws_2/image_display_pkg/image_display_pkg/ir/FP16/road-segmentation-adas-0001.xml'
        model_bin = '/root/deepracer_ws_2/image_display_pkg/image_display_pkg/ir/FP16/road-segmentation-adas-0001.bin'
        self.ie = IECore()
        self.net = self.ie.read_network(model=model_xml, weights=model_bin)
        self.exec_net = self.ie.load_network(network=self.net, device_name='CPU')

        # 입력 및 출력 노드 이름 가져오기
        self.input_blob = next(iter(self.net.input_info))
        self.output_blob = next(iter(self.net.outputs))

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 이미지 전처리
        preprocessed_image = self.preprocess_image(cv_image)

        cv2.namedWindow('Resized Result' , cv2.WINDOW_NORMAL)
        cv2.imshow('Resized Result', preprocessed_image[0].transpose((1,2,0)))
        cv2.waitKey(1)


        # 입력 데이터 생성
        input_data = {self.input_blob: preprocessed_image}

        # 추론 수행
        output = self.exec_net.infer(inputs=input_data)
    
        # 세그멘테이션 결과 처리
        segmentation_result = self.process_segmentation_output(output[self.output_blob])
            
        # 세그멘테이션 결과 표시
        cv2.namedWindow('Segmentation Result' , cv2.WINDOW_NORMAL)
        cv2.imshow('Segmentation Result', segmentation_result)
        cv2.waitKey(1)
        
    def preprocess_image(self, cv_image):
        image = cv_image
        
        ih, iw = image.shape[:-1]
        desired_height, desired_width = 512, 896
        
        # Resize the image while maintaining the aspect ratio
        scale = min(desired_height / ih, desired_width / iw)
        new_height = int(ih * scale)
        new_width = int(iw * scale)
        image = cv2.resize(image, (new_width, new_height))
        
        # Create a black canvas with the desired size
        canvas = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)
        
        # Calculate the offset to center the resized image on the canvas
        y_offset = (desired_height - new_height) // 2
        x_offset = (desired_width - new_width) // 2
        
        # Place the resized image on the canvas
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = image
        
        # Change data layout from HWC to CHW
        canvas = canvas.transpose((2, 0, 1))
       
        # Add batch dimension
        canvas = np.expand_dims(canvas, axis=0)

        return canvas

    def process_segmentation_output(self, output):
        
        # 클래스에 해당하는 색상 매핑
        class_colors = {
            0: [137, 195, 0],    # 배경 (B,G,R)
            1: [72, 63, 51],    # 도로
            2: [227, 232, 232],    # 트롤리
            3: [0, 163, 255]   # lane
        }
        
        height, width = output.shape[2:]
        segmentation_result = np.zeros((height, width, 3), dtype=np.uint8)

        
        for c in range(output.shape[1]):
            mask = (output[0, c, :, :] > 0.5)  # 클래스의 확률이 0.5보다 큰 픽셀 선택
            segmentation_result[mask] = class_colors[c]
         
        return segmentation_result


 
def main():
    rclpy.init()
    image_display = ImageDisplay()
    rclpy.spin(image_display)
    image_display.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
