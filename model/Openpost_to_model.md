## From the format of keypoint extracted from oprnpose 
```
T = number of frames,

76 = number of joints (face + hands + body),

3 = (x, y) coordinates + confidence score.
```

### Using the keypoints format above to pass into our model's preprocessing


```
Inferencing part of model is done:

The model takes in a shape of  T × 137 × 3 for one sentence. 
T = number of frames,

137 = number of joints keypoints (face + hands + body),

[3] = (x, y) coordinates, confidence score.
```
