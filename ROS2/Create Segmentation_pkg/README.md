# Create Segmentation_pkg  

## Result  

<p align="center">
<img width="700" src="../image/Create_Segmentation_pkg.png">
</p>
- Resized Result
<p align="center">
<img width="700" src="../image/Resized_Result.png">
</p> 
- Segmentation Result
<p align="center">
<img width="700" src="../image/Segmentation_Result.png">
</p> 
 

## Goal
**OpenVINO**에서 다운로드 받은 **road-segmentation-adas-0001** 모델을 로드하여 **camera_pkg**에서 받은 영상을 segmentation하여 **inference_pkg**로 전달

### How to make Segmentation_pkg

1. 이전에 만들어 둔 image_display_pkg를 그대로 이용하여 원본 영상과 Segmentation 된 영상을 비교할 예정   
2. OpenVINO의 모델 다운로더를 이용하여 model.yml 파일을 통해 .xml , .bin 파일을 다운로드 해주고 파일을 image_display_pkg에 ir 폴더로 이동    
3. OpenVINO 모델을 사용하기 위해서 IECore import  
```
from openvino.inference_engine import IECore
```

4. IR 모델 로드  
```
model_xml = '/root/deepracer_ws_2/image_display_pkg/image_display_pkg/ir/FP16/road-segmentation-adas-0001.xml'
        model_bin = '/root/deepracer_ws_2/image_display_pkg/image_display_pkg/ir/FP16/road-segmentation-adas-0001.bin'
        self.ie = IECore()
        self.net = self.ie.read_network(model=model_xml, weights=model_bin)
        self.exec_net = self.ie.load_network(network=self.net, device_name='CPU')
```
5. image_callback 함수에 과정을 구현  
```
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
```

6. 이미지 전처리, 후처리 함수를 구현  

```
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
```

