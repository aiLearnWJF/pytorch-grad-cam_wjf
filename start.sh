# 默认测试单张图片
# ~/miniconda3/envs/mmclassification/bin/python cam.py --image-path my_test/00195399_c1s1_1616070579620_01.jpg --method gradcam

# 自定义模型
# export PYTHONPATH=/home/yckj3860/code/moco_wjf/
# ~/miniconda3/envs/mmclassification/bin/python cam_wjf.py --image-path my_test/00000009_c0s1_4676_01.jpg --method gradcam

# 特征相似度可视化, 个人感觉这个不是用来看ebemding提取的，而是偏向与找相同或者找不同，不适合reid可视化， 需要改为在ebemding输出做损失计算
# ~/miniconda3/envs/mmclassification/bin/python PixelAttributionforembeddings.py

# 特征相似度可视化自定义模型，用于reid可视化，应该使用ebemding来求特征图的梯度，且同id用相似度，不同id用差异度; 感觉cos和1-cos表示的是相反的
~/miniconda3/envs/fast-reid/bin/python PixelAttributionforembeddings_fastReid_wjf.py

# vit注意力可视化
# ~/miniconda3/envs/mmclassification/bin/python vit_example_wjf.py --image-path my_test/00195289_c1s1_1616070326540_01.jpg --method gradcam