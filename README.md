# ML-NDT

- The custom CNN model trained on 796 ultrasonic training images and validated on 8 files achieved high generalization with a validation accuracy of up to 100%, outperforming pre-trained models in consistency and reliability.

- While ResNet50 reached 99.88% training accuracy, it struggled with generalization, showing only 46% validation accuracy. On the other hand, VGG19 demonstrated 98.34% training accuracy and 95% validation accuracy, but with significant oscillations in validation metrics.

- The final custom model achieved stable training accuracy (~93%), with Precision, Recall, and AUC closely tracking each other and steadily rising during training, confirming the model's robust and reliable performance over 100 epochs with 500 steps per epoch.
