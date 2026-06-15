# Pokemon-Visualization-Tool
A tool that recognizes pokemon in Pokemon Snap

Each file contains a model implementation, with the dataset generation file being contained in the data_operations folder.

The file Cross-Domain Object Detection and Classification contains a detailed report on the project completed by the listed authors.

Abstract:

Our project evaluates whether models trained only on synthetic 2D Pokemon sprite images can generalize to real´ 3D Pokemon Snap gameplay frames. We generated 10,000´ synthetic training images and evaluated on 165 manually labeled real game frames from the Beach level, containing 160 labeled Pokemon instances. We compared YOLOv8n,´ YOLO11n, and two stage YOLO + DINOv3 pipelines. Although the models learned the synthetic sprite domain well, they transferred poorly to real gameplay frames. The best end-to-end system, YOLO11n + DINOv3, correctly localized and classified only 12 of 160 real Pokemon instances.´ Error decomposition showed that detection was the main bottleneck: YOLO11n localized only 19 of 160 objects, while DINOv3 correctly classified 12 of those 19 detected crops. These results show that clean 2D synthetic training data is not sufficient for reliable detection in low resolution 3D game frames, and that future work should focus on reducing the synthetic to real localization gap.
