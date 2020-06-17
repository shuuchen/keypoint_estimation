# keypoint_estimation

### train
```
python train_regression.py --train-image-path  train.txt  --val-image-path val.txt --epochs 100 --batch-size 16
```
### test
```
python train_regression.py --task test --test-image-path test.txt --resume res/mse_channel/best_eval_model.pth.tar  --batch-size 1
```
