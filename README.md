Input Optimizer
===========
This is a library to optimize input image for any deep learning network based on a specific target, written in PyTorch `<https://pytorch.org/>`

## optimizing an input noise going through a pre-trained network on Mark1501 dataset.

### Samples of optimizing based on an specific attribute:

![c_5_iter_999_loss_-26 289833068847656](https://user-images.githubusercontent.com/89969561/184866464-f4bec1cd-e6c8-4bd4-a34d-2d74261edf3d.jpg) | ![c_30_iter_758_loss_-29 52212142944336](https://user-images.githubusercontent.com/89969561/184866376-db01073d-2371-4622-a934-202641562861.jpg) | ![c_25_iter_448_loss_-22 735559463500977](https://user-images.githubusercontent.com/89969561/184866334-963671f0-c7c4-429b-a2c1-5dafee5067cc.jpg)



## optimizing an input noise going through a pre-trained AlexNet model.
### Samples of optimizing based on an specific attribute:

<img src="https://user-images.githubusercontent.com/89969561/184868238-db5f0eee-bcf0-4631-97d8-bde43d05b73c.jpg" width="270"/> <img src="https://user-images.githubusercontent.com/89969561/184868271-456f9e6f-4ed2-423f-956d-a4ef90555e8d.jpg" width="270"/> <img src="https://user-images.githubusercontent.com/89969561/184868408-ae11166c-4d3d-4749-ac27-d78f68343310.jpg" width="270"/> 

Requirements
---------------
```python
    torch
    torchvision
    copy
    numpy
    PIL
