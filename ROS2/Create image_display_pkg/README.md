# Create image_display_pkg

## Goal

**sensor_fusion_pkg/overlay_msg를 받아서 opencv로 디스플레이하는 image_display_pkg를 만들기**

### How to make image_display_pkg

1.  현재 딥레이서 안에서 노드들 사이에서 통신하기 위해 노드들이 보내는 토픽 리스트를 확인

```python
ros2 topic list
```

2. 새로운 터미널에서 다음 명령어를 입력하여 ROS2 패키지를 생성합니다. 이때, 원하는 작업 디렉토리로 이동한 후에 실행

```python
ros2 pkg create --build-type ament_python image_display_pkg
```

3.  생성된 패키지 디렉토리로 이동

```python
cd image_display_pkg
```

4. 패키지에 필요한 종속성을 추가 **`setup.py`** 파일을 열고 **`install_requires`** 목록에 **`opencv-python`**을 추가

```python
from setuptools import setup

package_name = 'image_display_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    py_modules=[
        'image_display_pkg.image_display',
    ],
    install_requires=['setuptools', 'opencv-python'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Image display package for ROS2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_display = image_display_pkg.image_display:main',
        ],
    },
)
```

5. **`image_display_pkg/image_display_pkg`** 디렉토리 내에 **`image_display.py`** 파일을 생성하고 다음 코드를 작성 

```python
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageDisplay(Node):
    def __init__(self):
        super().__init__('image_display')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/sensor_fusion_pkg/overlay_msg',
            self.image_callback,  # 쉼표 제거
            10)
        self.subscription

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('Image Display', cv_image)
        cv2.waitKey(1)

def main():
    rclpy.init()
    image_display = ImageDisplay()
    rclpy.spin(image_display)
    image_display.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

6. **`image_display_pkg/image_display_pkg`** 디렉토리에 있는 **`__init__.py`** 파일을 열고, 다음 코드를 추가

```python
from .image_display import ImageDisplay
```

7. 패키지를 설치, 패키지 루트 디렉토리에서 다음 명령어를 실행

```python
source /opt/ros/foxy/setup.bash  # ROS2 버전에 맞게 경로를 변경하세요.
colcon build --symlink-install
```

8. 생성된 패키지를 실행하려면, 다음 명령어를 입력

```python
source install/setup.bash
ros2 run image_display_pkg image_display
```
