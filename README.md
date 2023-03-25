# 数据集说明

Created: March 24, 2023 5:22 PM

1. aidatatang_200zh：这是一个包含 200 小时中文语音数据的数据集，用于语音识别任务。
    - 数据格式：音频文件（.wav）和对应的文本标注（.txt）
    - 下载地址：**[http://www.openslr.org/62/](http://www.openslr.org/62/)**
    
    ```python
    # PyTorch 和 TensorFlow:
    暂无官方提供的预处理方法，请自行处理音频和文本数据。
    ```
    
2. FlyingChairs：这是一个用于计算机视觉中的光流估计任务的数据集，包含了 22,872 对图像。
    - 数据格式：图像对（.png）
    - 下载地址：**[https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip](https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip)**
    
    ```python
    # PyTorch 和 TensorFlow:
    暂无官方提供的预处理方法，请自行处理图像对和光流数据。
    ```
    
3. OASIS：Open Access Series of Imaging Studies 这是一个用于认知能力研究的 MRI 大脑成像数据集，包含了 416 个受试者的 MRI 图像和临床数据。
    - 数据格式：MRI图像（.nii）和临床数据（.csv）
    - 下载地址：需要在OASIS项目网站上进行注册并申请数据访问权限。网址：**[https://www.oasis-brains.org/](https://www.oasis-brains.org/)**
    
    ```python
    # PyTorch 和 TensorFlow:
    暂无官方提供的预处理方法，请自行处理MRI图像和临床数据。
    ```
    
4. ST-CMDS-20170001_1-OS：这是一个包含 10,000 个普通话音频文件和对应的文本标注的中文语音识别数据集。
    - 数据格式：音频文件（.wav）和对应的文本标注（.txt）
    - 下载地址：**[http://www.openslr.org/38/](http://www.openslr.org/38/)**
    
    ```python
    # PyTorch 和 TensorFlow:
    暂无官方提供的预处理方法，请自行处理音频和文本数据。
    ```
    
5. AISHELL：这是一个包含 170 小时普通话语音数据和对应文本标注的中文语音识别数据集。
    - 数据格式：音频文件（.wav）和对应的文本标注（.txt）
    - 下载地址：**[http://www.openslr.org/33/](http://www.openslr.org/33/)**
    
    ```python
    # PyTorch:
    from torchaudio.datasets import AISHELL
    aishell_dataset = AISHELL(root="data_path", download=True)
    
    # TensorFlow:
    import tensorflow_io as tfio
    aishell_dataset = tfio.IOTensor.from_audio("data_path/*.wav")
    ```
    
6. Glint360K：这是一个用于人脸识别任务的数据集，包含超过 36 万张图像和标注。
    - 数据格式：人脸图像（.jpg）和标注文件（.txt）
    - 下载地址：该数据集需要提交申请，申请表格和相关信息请查看Glint360K项目主页：**[https://github.com/deepinsight/insightface/wiki/Glint360K](https://github.com/deepinsight/insightface/wiki/Glint360K)**
    
    ```python
    # PyTorch 和 TensorFlow:
    暂无官方提供的预处理方法，请自行处理图像和人脸识别标注数据。
    ```
    
7. OpenImages Dataset：这是一个用于计算机视觉中目标检测和图像分类任务的数据集，包含超过 900 万张图像和标注。
    - 数据格式：图像文件（.jpg）和标注文件（.csv）
    - 下载地址：OpenImagesDataset提供了多种不同数据子集和标注类型。您可以访问其官方GitHub仓库获取详细信息并下载所需数据：**[https://github.com/openimages/dataset](https://github.com/openimages/dataset)**
    
    ```python
    # PyTorch:
    from torchvision.datasets import OpenImages
    openimages_dataset = OpenImages(root="data_path", split="train", download=True)
    
    # TensorFlow:
    import tensorflow_datasets as tfds
    openimages_dataset = tfds.load("open_images_v4", split="train", data_dir="data_path")
    ```
    
8. TEDLIUM_release-3：这是一个包含了超过 450 小时的 TED 演讲音频数据和对应的文本标注的数据集，用于语音识别任务。
    - 数据格式：音频文件（.sph）和对应的文本标注（.stm）
    - 下载地址：**[http://www.openslr.org/51/](http://www.openslr.org/51/)**
    
    ```python
    # PyTorch:
    import torchaudio
    train_dataset = torchaudio.datasets.TEDLIUM("data_path", release=3, subset="train", download=True)
    
    # TensorFlow:
    暂无官方提供的预处理方法，请自行处理音频和文本数据。
    ```
    
9. AlphaFold2：这是一个用于蛋白质结构预测任务的数据集，包含了大量蛋白质序列和对应的结构信息。
    - 数据格式：蛋白质序列（.fasta）和对应的结构信息（.pdb）
    - 下载地址：Alphafold2是一个模型，而不是一个数据集。您可以访问其GitHub仓库了解更多信息并下载源代码：**[https://github.com/deepmind/alphafold](https://github.com/deepmind/alphafold)**
    
    ```python
    # PyTorch 和 TensorFlow：
    参考 AlphaFold2 的官方代码，使用对应的深度学习框架进行蛋白质结构预测任务。
    ```
    
10. GoPro：这是一个用于计算机视觉中的运动去模糊任务的数据集，包含了高速摄像机拍摄的 32 个场景的图像序列。
    - 数据格式：高速摄像机拍摄的图像序列（.png）
    - 下载地址：**[https://github.com/SeungjunNah/DeepDeblur_release/tree/master/dataset](https://github.com/SeungjunNah/DeepDeblur_release/tree/master/dataset)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理运动去模糊相关的图像数据。
    ```
    
11. Open Images V6 MLPerf：这是一个用于计算机视觉中目标检测和图像分类任务的数据集，包含了超过 190 万张图像和标注。
    - 数据格式：图像文件（.jpg）和标注文件（.csv）
    - 下载地址：**[https://storage.googleapis.com/openimages/web/download.html](https://storage.googleapis.com/openimages/web/download.html)**
    
    ```python
    # PyTorch:
    # 请注意，PyTorch不提供专门用于Open Images数据集的预处理方法。您需要自行处理数据集。
    # from torchvision.datasets import VOCDetection
    # openimages_dataset = VOCDetection(root="data_path", year="2019", image_set="train", download=True)
    
    # TensorFlow:
    import tensorflow_datasets as tfds
    openimages_dataset, info = tfds.load("open_images_v6", with_info=True, data_dir="data_path")
    ```
    
12. CIFAR：这是一个用于计算机视觉中图像分类任务的数据集，包含了 60,000 张 32x32 像素的图像和标注，分为 10 类。
    - 数据格式：图像文件（.png）和标注文件（.csv）
    - 下载地址：**[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)**
    
    ```python
    # PyTorch:
    import torchvision.datasets as datasets
    train_dataset = datasets.CIFAR10(root="data_path", train=True, download=True)
    test_dataset = datasets.CIFAR10(root="data_path", train=False, download=True)
    
    # TensorFlow:
    import tensorflow_datasets as tfds
    cifar_dataset, info = tfds.load("cifar10", with_info=True, data_dir="data_path")
    ```
    
13. ImageNet2012：这是一个用于计算机视觉中图像分类任务的数据集，包含了 100 万张高分辨率图像和标注，分为 1000 类。
    - 数据格式：图像文件（.JPEG）和标注文件（.xml）
    - 下载地址：**[http://image-net.org/challenges/LSVRC/2012/index](http://image-net.org/challenges/LSVRC/2012/index)**
    
    ```python
    # PyTorch:
    # 请注意，PyTorch不提供专门用于ImageNet数据集的预处理方法。您需要自行处理数据集
    
    # TensorFlow:
    import tensorflow_datasets as tfds
    imagenet_dataset, info = tfds.load("imagenet2012", with_info=True, data_dir="data_path")
    ```
    
14. PASCAL VOC：这是一个用于计算机视觉中目标检测和图像分类任务的数据集，包含了 20 个物体类别的 17,125 张图像和标注。
    - 数据格式：图像文件（.JPEG）和标注文件（.xml）
    - 下载地址：**[http://host.robots.ox.ac.uk/pascal/VOC/](http://host.robots.ox.ac.uk/pascal/VOC/)**
    
    ```python
    # PyTorch:
    from torchvision.datasets import VOCDetection
    voc_dataset = VOCDetection(root="data_path", year="2012", image_set="train", download=True)
    
    # TensorFlow:
    import tensorflow_datasets as tfds
    voc_dataset, info = tfds.load("voc/2012", with_info=True, data_dir="data_path")
    ```
    
15. TrackingNet：这是一个用于目标跟踪任务的数据集，包含了超过 30,000 个跟踪序列。
    - 数据格式：视频文件（.mp4）和标注文件（.txt）
    - 下载地址：**[https://tracking-net.org/](https://tracking-net.org/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理目标跟踪相关的数据。
    ```
    
16. BDD100K：这是一个用于自动驾驶任务的数据集，包含了超过 10 万张图像和标注。
    - 数据格式：图像文件（.jpg）和标注文件（.json）
    - 下载地址：**[https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理自动驾驶相关的图像和标注数据。
    ```
    
17. PDB100：这是一个用于蛋白质结构预测任务的数据集，包含了 100 个蛋白质的结构信息。
    - 数据格式：蛋白质结构文件（.pdb）
    - 下载地址：PDB100 数据集是用于蛋白质结构预测的一个挑战赛的数据集。详细信息和数据集可在挑战赛官网找到：**[https://predictioncenter.org/casp14/index.cgi](https://predictioncenter.org/casp14/index.cgi)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理蛋白质结构预测相关的数据。
    ```
    
18. THCHS-30：这是一个用于中文语音识别任务的数据集，包含了 30 小时的中文语音数据和对应的文本标注。
    - 数据格式：音频文件（.wav）和对应的文本标注（.trn）
    - 下载地址：**[http://www.openslr.org/18/](http://www.openslr.org/18/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理中文语音识别相关的音频和文本数据。
    ```
    
19. UniClust30：这是一个用于蛋白质序列分析的数据集，包含了超过 6 万个蛋白质序列。
    - 数据格式：蛋白质序列文件（.fasta）
    - 下载地址：**[http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/](http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理蛋白质序列分析相关的数据。
    ```
    
20. ImageNet Mini：这是 imagenet2012 数据集的一个子集，包含了 13 万张图像和标注，用于快速原型开发和测试。
    - 数据格式：图像文件（.JPEG）和标注文件（.xml）
    - 下载地址：ImageNet-mini 数据集没有官方下载链接。您可以尝试从第三方资源下载，但要确保数据集来源可靠并遵守相关许可和使用条款。
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理图像分类相关的图像和标注数据。
    ```
    
21. PDB_MMCIF：这是一个用于蛋白质结构预测任务的数据集，包含了超过 180,000 个蛋白质的结构信息。
    - 数据格式：蛋白质结构文件（.cif）
    - 下载地址：**[https://www.rcsb.org/](https://www.rcsb.org/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理蛋白质结构预测相关的数据。
    ```
    
22. UniRef30_2020_06：这是一个用于蛋白质序列分析的数据集，包含了超过 50 亿个非冗余蛋白质序列。
    - 数据格式：蛋白质序列文件（.fasta）
    - 下载地址：**[https://www.uniprot.org/uniref/](https://www.uniprot.org/uniref/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理蛋白质序列分析相关的数据。
    ```
    
23. Ceramic：这是一个用于玻璃陶瓷缺陷检测任务的数据集，包含了 4,500 张图像和标注。
    - 数据格式：图像文件（.jpg或.png）和标注文件（.json或.xml）
    - 下载地址：该数据集未找到官方下载链接，请查找可靠的第三方资源。
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理玻璃陶瓷缺陷检测相关的图像和标注数据。
    ```
    
24. InterHand：这是一个用于手部姿态估计任务的数据集，包含了 14,817 张图像和标注。
    - 数据格式：图像文件（.jpg）和标注文件（.json）
    - 下载地址：**[https://github.com/facebookresearch/InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理手部姿态估计相关的图像和标注数据。
    ```
    
25. Places365：这是一个用于场景分类任务的数据集，包含了超过 180 万张图像和标注，分为 365 类场景。
    - 数据格式：图像文件（.jpg）和标注文件（.txt）
    - 下载地址：**[http://places2.csail.mit.edu/download.html](http://places2.csail.mit.edu/download.html)**
    
    ```python
    # PyTorch:
    from torchvision.datasets import Places365
    places365_dataset = Places365(root="data_path", split="train-standard", download=True)
    
    # TensorFlow:
    暂无官方提供的预处理方法，请自行处理场景分类相关的图像和标注数据。
    ```
    
26. UniRef90：这是一个用于蛋白质序列分析的数据集，包含了超过 90 亿个非冗余蛋白质序列。
    - 数据格式：蛋白质序列文件（.fasta）
    - 下载地址：**[https://www.uniprot.org/uniref/](https://www.uniprot.org/uniref/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理蛋白质序列分析相关的数据。
    ```
    
27. COCO2014：这是一个用于目标检测、图像分割和关键点检测任务的数据集，包含了超过 330,000 张图像和标注，分为 80 类目标。
    - 数据格式：图像文件（.jpg）和标注文件（.json）
    - 下载地址：**[https://cocodataset.org/#download](https://cocodataset.org/#download)**
    
    ```python
    # PyTorch:
    from torchvision.datasets import CocoDetection
    coco_dataset = CocoDetection(root="data_path/images", annFile="data_path/annotations/instances_train2014.json", download=True)
    
    # TensorFlow:
    import tensorflow_datasets as tfds
    coco_dataset, info = tfds.load("coco/2014", with_info=True, data_dir="data_path")
    ```
    
28. LibriSpeech：这是一个用于语音识别任务的数据集，包含了超过 1,000 小时的英语语音数据和对应的文本标注。

     - 数据格式：音频文件（.flac）和对应的文本标注（.txt）
     - 下载地址：**[http://www.openslr.org/12/](http://www.openslr.org/12/)**
    
    ```python
    # PyTorch:
    import torchaudio
    train_dataset = torchaudio.datasets.LIBRISPEECH(root="data_path", url="train-clean-100", download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(root="data_path", url="test-clean", download=True)  
    
    # TensorFlow:
    import tensorflow_datasets as tfds
    librispeech_dataset, info = tfds.load("librispeech_clean", with_info=True, data_dir="data_path")
    ```
    
29. PrimeWords_MD_2018_Set1：这是一个用于语音识别任务的数据集，包含了超过 10 小时的普通话语音数据和对应的文本标注。
    - 数据格式：音频文件（.wav）和对应的文本标注（.txt）
    - 下载地址：**[http://www.openslr.org/47/](http://www.openslr.org/47/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理语音识别相关的音频和文本数据。
    ```
    
30. Semantic3D：这是一个用于点云分类和分割任务的数据集，包含了超过 4 亿个点云和标注。
    - 数据格式：点云数据文件（.txt或.las）和标注文件（.labels）
    - 下载地址：**[http://www.semantic3d.net/](http://www.semantic3d.net/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理点云分类和分割相关的数据。
    ```
    
31. KITTI-2012：这是一个用于自动驾驶任务的数据集，包含了超过 42,000 张图像和标注。
    - 数据格式：图像文件（.png）、点云数据文件（.bin）和标注文件（.txt）
    - 下载地址：**[http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理自动驾驶相关的图像和标注数据。
    ```
    
32. PR2K：这是一个用于自动驾驶任务的数据集，包含了 5,000 张图像和标注。
    - 数据格式：图像文件（.jpg或.png）和标注文件（.json或.xml）
    - 下载地址：该数据集未找到官方下载链接，请查找可靠的第三方资源。
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理自动驾驶相关的图像和标注数据。
    ```
    
33. VCTK-Corpus：这是一个用于语音识别任务的数据集，包含了英式英语口音的语音数据和对应的文本标注。
    - 数据格式：音频文件（.flac或.wav）和对应的文本标注（.txt）
    - 下载地址：**[http://www.openslr.org/92/](http://www.openslr.org/92/)**
    
    ```python
    # PyTorch:
    from torchaudio.datasets import VCTK
    vctk_dataset = VCTK(root="data_path", download=True)
    
    # TensorFlow:
    暂无官方提供的预处理方法，请自行处理语音识别相关的音频和文本数据。
    ```
    
34. coco2017：这是 coco2014 数据集的更新版本，包含了超过 330,000 张图像和标注，分为 80 类目标。
    - 数据格式：图像文件（.jpg）和标注文件（.json）
    - 下载地址：**[https://cocodataset.org/#download](https://cocodataset.org/#download)**
    
    ```python
    # PyTorch:
    from torchvision.datasets import CocoDetection
    coco_dataset = CocoDetection(root="data_path/images", annFile="data_path/annotations/instances_train2017.json", download=True)
    
    # TensorFlow:
    import tensorflow_datasets as tfds
    coco_dataset, info = tfds.load("coco/2017", with_info=True, data_dir="data_path")
    ```
    
35. RP2K_RP2K_Dataset：这是一个用于自动驾驶任务的数据集，包含了超过 2,000 张图像和标注。
    - 数据格式：图像文件（.jpg或.png）和标注文件（.json或.xml）
    - 下载地址：该数据集未找到官方下载链接，请查找可靠的第三方资源。
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理自动驾驶相关的图像和标注数据。
    ```
    
36. Vimeo-90k：这是一个用于视频去噪任务的数据集，包含了超过 90,000 个视频片段。
    - 数据格式：视频文件（.mp4）
    - 下载地址：**[http://toflow.csail.mit.edu/](http://toflow.csail.mit.edu/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理视频去噪相关的视频片段数据。
    ```
    
37. CULane：这是一个用于自动驾驶任务的数据集，包含了 55 小时的道路场景视频和标注。
    - 数据格式：图像文件（.jpg）和标注文件（.lines）
    - 下载地址：**[https://xingangpan.github.io/projects/CULane.html](https://xingangpan.github.io/projects/CULane.html)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理自动驾驶相关的道路场景视频和标注数据。
    ```
    
38. MNIST：这是一个经典的用于图像分类任务的数据集，包含了 60,000 张 28x28 像素的手写数字图像和标注。
    - 数据格式：图像文件（.idx3-ubyte）和标签文件（.idx1-ubyte）
    - 下载地址：**[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)**
    
    ```python
    # PyTorch:
    from torchvision.datasets import MNIST
    mnist_dataset = MNIST(root="data_path", train=True, download=True)
    
    # TensorFlow:
    import tensorflow_datasets as tfds
    mnist_dataset, info = tfds.load("mnist", with_info=True, data_dir="data_path")
    ```
    
39. ShapeNetCore.v2：这是一个用于三维物体分类、分割和形状生成任务的数据集，包含了超过 50 万个三维物体模型和标注。
    - 数据格式：三维模型文件（.obj）和标注文件（.txt）
    - 下载地址：**[https://www.shapenet.org/](https://www.shapenet.org/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理三维物体分类、分割和形状生成相关的三维物体模型和标注数据。
    ```
    
40. VoxCeleb：这是一个用于语音合成任务的数据集，包含了超过 1,000 小时的英语语音数据和对应的文本标注。
    - 数据格式：音频文件（.m4a）和对应的文本标注（.txt）
    - 下载地址：**[https://www.robots.ox.ac.uk/~vgg/data/voxceleb/](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理语音合成相关的音频和文本数据。
    ```
    
41. MIMIC-CXR：这是一个用于医疗图像分析任务的数据集，包含了超过 370,000 张 X 光胸部图像和标注。
    - 数据格式：X光胸部图像文件（.jpg或.dcm）和标注文件（.csv）
    - 下载地址：**[https://physionet.org/content/mimic-cxr/2.0.0/](https://physionet.org/content/mimic-cxr/2.0.0/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理医疗图像分析相关的 X 光胸部图像和标注数据。
    ```
    
42. nuScenes：这是一个用于自动驾驶任务的数据集，包含了超过 1,000 小时的高分辨率 3D 激光雷达数据和标注。
    - 数据格式：激光雷达数据文件（.pcd或.bin）、图像文件（.jpg）和标注文件（.json）
    - 下载地址：**[https://www.nuscenes.org/download](https://www.nuscenes.org/download)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理自动驾驶相关的高分辨率 3D 激光雷达数据和标注数据。
    ```
    
43. ST-AEDS-20180100：这是一个用于语音识别任务的数据集，包含了英式英语口音和美式英语口音的语音数据和对应的文本标注。
    - 数据格式：音频文件（.wav）和对应的文本标注（.txt）
    - 下载地址：**[http://www.openslr.org/85/](http://www.openslr.org/85/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理语音识别相关的英式英语口音和美式英语口音的语音数据和文本标注。
    ```
    
44. FlyingChairs：这是一个用于计算机视觉中的光流估计任务的数据集，包含了 22,872 对图像。
    - 数据格式：图像文件（.png）和光流估计标注文件（.flo）
    - 下载地址：**[https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html)**
    
    ```python
    # PyTorch 和 TensorFlow
    暂无官方提供的预处理方法，请自行处理计算机视觉中光流估计任务相关的图像数据。
    ```
    
46. TIMIT：这是一个用于语音识别任务的数据集，包含了美式英语和加拿大法语的语音数据和对应的文本标注。
    - 数据格式：音频文件（.wav）和对应的文本标注（.txt）
    - 下载地址：**[https://catalog.ldc.upenn.edu/LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1)**
    
    ```python
    # PyTorch
    import torchaudio
    train_dataset = torchaudio.datasets.TIMIT("data_path", url="train", download=True)
    
    # TensorFlow
    暂无官方提供的预处理方法，请自行处理音频和文本数据。
    ```
    
47. CommonVoice：这是一个用于语音识别任务的数据集，包含了多种语言的语音数据和对应的文本标注。
    - 数据格式：音频文件（.mp3）和对应的文本标注（.txt）
    - 下载地址：**[https://commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets)**
    
    ```python
    # PyTorch
    import torchaudio
    train_dataset = torchaudio.datasets.COMMONVOICE("data_path", url="en", download=True)
    
    # TensorFlow
    暂无官方提供的预处理方法，请自行处理音频和文本数据。
    ```
48. ADE20K：这是一个用于场景解析任务的数据集，包含了 2 万张图像及其对应的标注。
    - 数据格式：图像文件（.jpg）和标注文件（.png）
    - 下载地址：http://groups.csail.mit.edu/vision/datasets/ADE20K/
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理场景解析相关的图像和标注数据。
    ```
49. 3DShapes：这是一个用于生成模型任务的数据集，包含了 480,000 张 3D 形状的图像。
    
     - 数据格式：图像文件（.png）
     - 下载地址：**[https://github.com/deepmind/3dshapes-dataset](https://github.com/deepmind/3dshapes-dataset)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理 3D 形状的图像数据。
    ```
    
50. Reddit TIFU：这是一个用于自然语言处理任务的数据集，包含了 Reddit 上 TIFU 版块的帖子。
    
     - 数据格式：文本文件（.jsonl）
     - 下载地址：**[https://github.com/ryankiros/reddit-tifu](https://github.com/ryankiros/reddit-tifu)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理 Reddit TIFU 版块的帖子数据。
    ```
    
51. WikiText-103：这是一个用于自然语言处理任务的数据集，包含了维基百科上的文章。
    
     - 数据格式：文本文件（.txt）
     - 下载地址：**[https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)**
    
    ```python
    # PyTorch: 
    import torchtext.datasets as datasets 
    train_dataset, valid_dataset, test_dataset = datasets.WikiText103(root="data_path", split=("train", "valid", "test"))  
    
    # TensorFlow: 
    import tensorflow_datasets as tfds wikitext_dataset, info = tfds.load("wikitext103", with_info=True, data_dir="data_path")
    ```
    
52. Amazon Reviews：这是一个用于情感分析任务的数据集，包含了亚马逊网站上的评论。
    
     - 数据格式：文本文件（.json.gz）
     - 下载地址：**[http://jmcauley.ucsd.edu/data/amazon/](http://jmcauley.ucsd.edu/data/amazon/)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理亚马逊网站上的评论数据。
    ```
    

53. Yelp Reviews：这是一个用于情感分析任务的数据集，包含了 Yelp 网站上的评论。
     - 数据格式：文本文件（.json）
     - 下载地址：**[https://www.yelp.com/dataset](https://www.yelp.com/dataset)**
    
    ```python
    # PyTorch 和 TensorFlow：
    暂无官方提供的预处理方法，请自行处理 Yelp 网站上的评论数据。
    ```
    
55. SQuAD：这是一个用于阅读理解任务的数据集，包含了大量维基百科文章及问题答案对。
     - 数据格式：文本文件（.json）
     - 下载地址：**[https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)**

    ```python
    # PyTorch: 
    import transformers from datasets import load_dataset squad_dataset = load_dataset("squad")  
    
    # TensorFlow: 
    import tensorflow_datasets as tfds squad_dataset, info = tfds.load("squad", with_info=True, data_dir="data_path")
    ```
    
56. MS COCO：这是一个用于计算机视觉任务（如目标检测、图像分割等）的数据集，包含了 20 万张图像及其对应的标注。
     - 数据格式：图像文件（.jpg）和标注文件（.json）
     - 下载地址：**[https://cocodataset.org/#download](https://cocodataset.org/#download)**

    ```python
    # PyTorch: 
    import torchvision.datasets as datasets coco_dataset = datasets.CocoDetection(root="data_path/images", annFile="data_path/annotations/instances_train2017.json", download=True)  
    
    # TensorFlow: 
    import tensorflow_datasets as tfds coco_dataset, info = tfds.load("coco", with_info=True, data_dir="data_path")```
    
57. TACRED：这是一个用于关系抽取任务的数据集，包含了大量的实体关系标注。
     - 数据格式：文本文件（.json）
     - 下载地址：**[https://nlp.stanford.edu/projects/tacred/](https://nlp.stanford.edu/projects/tacred/)**
    
    ```python
    # PyTorch 和 TensorFlow： 
    由于 TACRED 数据集的访问受限，需要注册并获取许可后才能下载。因此，暂无官方提供的预处理方法。请在获得许可后自行处理 TACRED 数据集中的实体关系标注数据。
    ```


58. LFW (Labeled Faces in the Wild)：这是一个用于人脸识别和人脸验证任务的数据集，包含了 13,000 张名人照片及其对应的标注。
    - 数据格式：图像文件（.jpg）和对应的文本标注（.txt）
    - 下载地址：**[http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)**
    
    ```
    # PyTorch: 
    import torchvision.datasets as datasets lfw_dataset = datasets.LFW(root="data_path", download=True) 
    
    # TensorFlow: 
    import tensorflow_datasets as tfds lfw_dataset, info = tfds.load("lfw", with_info=True, data_dir="data_path")
    ```
    
59. IEMOCAP：这是一个用于语音情感识别任务的数据集，包含了大量多模态对话数据，如音频、视频和文本。
    - 数据格式：音频文件（.wav）、视频文件（.mp4）和对应的文本标注（.txt）
    - 下载地址：**[https://sail.usc.edu/iemocap/](https://sail.usc.edu/iemocap/)**
    
    ```
    # PyTorch 和 TensorFlow： 
    暂无官方提供的预处理方法，请自行处理 IEMOCAP 数据集中的音频、视频和文本数据。
    ```
    
60. VCTK：这是一个用于语音合成任务的数据集，包含了多个讲者的英语朗读音频及对应的文本标注。
    - 数据格式：音频文件（.wav）和对应的文本标注（.txt）
    - 下载地址：**[http://www.udialogue.org/download/VCTK-Corpus.tar.gz](http://www.udialogue.org/download/VCTK-Corpus.tar.gz)**
    
    ```
    # PyTorch 和 TensorFlow： 
    暂无官方提供的预处理方法，请自行处理 VCTK 数据集中的音频和文本数据。
    ```

61. CelebA (Large-scale CelebFaces Attributes)：这是一个用于人脸属性识别任务的数据集，包含了 200,000 张名人照片及其 40 个属性标注。
     - 数据格式：图像文件（.jpg）和对应的属性标注文件（.txt）
     - 下载地址：**[http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)**
    
    ```python
    # PyTorch: 
    import torchvision.datasets as datasets celeba_dataset = datasets.CelebA(root="data_path", download=True)  
    
    # TensorFlow: 
    import tensorflow_datasets as tfds celeba_dataset, info = tfds.load("celeb_a", with_info=True, data_dir="data_path")
    ```

    
62. Cityscapes：这是一个用于语义分割任务的数据集，包含了 5,000 张高分辨率城市场景图像及其对应的像素级标注。
     - 数据格式：图像文件（.png）和对应的像素级标注文件（.png）
     - 下载地址：**[https://www.cityscapes-dataset.com/](https://www.cityscapes-dataset.com/)**

    ```python
    # PyTorch 和 TensorFlow： 
    暂无官方提供的预处理方法，请自行处理 Cityscapes 数据集中的图像和像素级标注数据。
    ```

