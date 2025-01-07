# 2024-02-CSC4004-3-3-Team3
: Open Source Project Class, Dongguk University, Seoul
- Team Leader : JunKyu Park
- Team Members : Chanyoung Lee, Eunseo Kim, Junghyun Kim

## Creating Bounding Boxes on 3D Map

### abstract
This project leverages robotics, deep learning-based crack detection, and 3D mapping technologies to automate indoor inspections and present results through an intuitive, interactive, web-based 3D map viewer. The robot collects data using high-resolution cameras and sensors, while a deep learning model ensures fast and accurate detection of cracks and defects. The collected data is processed into a high-quality 3D point cloud map using COLMAP, Gaussian splatting, and 3D rendering techniques. These results are visualized on a web platform, enabling real-time inspection and efficient maintenance planning. This technology is also applicable to various industries, including construction, facility management, and disaster response.

### implementation
Please set up an environment identical to the one provided by the two GitHub pages below.




input 
1. Output of Colmap from the image stream robot takes
2. Output of 2D Crack Detection model ( JSON format )

<details>
<summary>Click to toggle contents of detail JSON Format</summary>

```

```
</details>

</br>

Result
Image stream from Robot</br>
![c](https://github.com/user-attachments/assets/c4656cc0-2e3e-4f73-ac0d-129ed3e1d907)

Image stream from handheld Camera ( iphone 16-pro )

![a](https://github.com/user-attachments/assets/2e298659-1943-4c00-af92-a8b5be8fe609)


![b](https://github.com/user-attachments/assets/a34dc340-e64a-4814-a96c-bdc89cc433ec)
