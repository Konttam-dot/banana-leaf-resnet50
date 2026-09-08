# Banana Leaf Classification using ResNet50

โปรเจกต์จำแนกประเภทใบกล้วยด้วย Deep Learning
โดยใช้โมเดล ResNet50

## Model
- ResNet50
- Transfer Learning
- Input Size: 224x224
- Optimizer: Adam
- Learning Rate: 0.0001
- Epochs: 20

## Dataset
ใช้ Dataset ใบกล้วย แบ่งเป็น
- Train
- Validation
- Test

## Evaluation
ประเมินโมเดลด้วย
- Accuracy
- Loss
- Confusion Matrix
- Precision
- Recall
- F1-score

## Files
- `best_resnet50.ipynb` - Notebook สำหรับ Train และทดสอบโมเดล
- `best_resnet50_banana.pth` - โมเดล ResNet50 ที่ผ่านการ Train
