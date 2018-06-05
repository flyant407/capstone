# Machine Learning Engineer Nanodegree
## Capstone Proposal
Baijie  
June 2nd, 2018

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.
本项目是美国的汽车保险和金融公司State Farm 2年前在kaggle上发起的一个比赛，主题是“Can computer vision spot distracted drivers?”。根据CDC汽车安全部门统计，五分之一的车祸是由司机的分心造成的，也就是说每年因分心导致交通事故死亡的人数有3000人。State Farm提供了通过在汽车仪表盘上的摄像机获取到司机驾驶过程中的2D图像作为数据集，来给每个驾驶员的行为作出分类。
该项目的图像是在一个可控的实验条件下得到的。司机并没有实际的在控制车辆，车辆是由一辆卡车拉着在路上行使的。
目前深度学习在图像识别领域表现出了巨大的威力，

### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).


### Datasets and Inputs
_(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.
本项目中提供的数据集是司机驾驶过程中的照片，有的在发信息、吃东西，打电话或者是专心开车等。已经给出了司机可能的10种状态，如下：
- c0: safe driving
- c1: texting - right
- c2: talking on the phone - right
- c3: texting - left
- c4: talking on the phone - left
- c5: operating the radio
- c6: drinking
- c7: reaching behind
- c8: hair and makeup
- c9: talking to passenger

我们的目标是识别出每幅图片中司机的状态。
一个司机的照片只会出现在训练数据集或测试数据集之中的一个。
### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).
本项目准备采用巻积神经网络(CNN)来实现该功能。巻积运算是神经网络最基本的组成部分，这里的巻积与数学上的巻积不同，数学中巻积的定义是先将矩阵做镜像反转，之后在作元素乘积求和，而在这里的巻积运算只是是将filter与元图像矩阵中的数据乘积之后加权求和。巻积运算能够在CNN中能够在图像处理领域取得非常好的效果的主要原因是CNN具有参数共享和稀疏连接的特性。参数共享即同一个filter在整个图像上训练出来的参数只有一份，实现了图像的平移不变的特性。同时也减小了模型的参数数量，便于对高清图像的处理。稀疏连接指的是CNN输出层中的每个元素只依赖于输入层中和filter做巻积的那几个元素，输入层中的其他元素都不会影响该输出层中的元素。CNN通过这两种机制减少参数，以便我们用更小的训练集来训练它，从而预防过拟合。
经典的CNN网络介绍如下：
> 1. LeNet-5：主要是针对灰度图片来训练的，现在已经很少使用了，但是其提出的网络结构至今仍被大量使用：一个或多个卷积层后面跟一个池化层，然后又是若干个巻积层在接一个池化层，然后是全链接层,最后是输出 。
> 2. AlexNet网络,与LeNet-5相比，其结构相似，但是AlexNet网络更大，并且AlexNet采用了Relu激活函数代替了LeNet-5使用的sigmod激活函数。
> 3. VGG网络是一种专注于构建巻积层的简单网络，结构的优点是：结构规整，巻积层的数量是翻倍增长，而池化后的图像长、宽都减少一半。其缺点是需要训练的参数巨大。VGG16网络包含16个巻积层和全链接层，总共含有约1.38亿个参数，VGG19比VGG16的网络更大。虽然VGG19比VGG16的网络结构更大，但是表现却没有比VGG16更好。
> 4. Google的Inception网络；该网络带来了一些结构上的改变。不需要人为的决定使用哪个过滤器或者是否需要池化，而是由网络自行确定这些参数。我们可以给网络添加这些参数的所有可能值，然后把这些输出连起来，让网络自己学习它需要什么杨的参数，采用哪些filter的组合。该网络的优点是代替了人工选择网络结构。但是由于我们加入了太多的结构，计算成本变得巨大。这里引入了1x1巻积，得到一个bottleneck layer，在可以有效减少计算成本的同时还可以给网络增加非线性函数。
> 5. ResNets：由于存在梯度消失和梯度爆炸的问题，非常深的网络是很难训练的，ResNets采用跳跃连接构，可以构建超过100层的ResNets网络。
对于计算机视觉的问题的处理，如果从头开始训练一个网络的权重，一方面需要巨大的计算成本（包括时间和金钱），另一方面也需要大量的训练样本。在本项目中我既没有能够训练上亿个参数的训练样本（训练集只有22424个样本），也没有训练的条件（使用的是aws免费的EC2 p2.xlarge 实例,该实例有4个CPU,60GB内存,1个NVIDIA K80 GPU,12GB GPU内存。机器系统为 Ubuntu 18.04.5 LTS)。所以这里使用迁移学习。采用keras里面已经训练好的模型。这些模型使用的数据集是ImageNet。在本项目中我将把他们当作一个很好的初始化用在我自己的神经网络中，站在前人们的肩膀上，好好的利用他们的成果。

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.


### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).
这里采用kaggle上该项目的要求，提交到kaggle上的是每张图片被分类到每个类别的概率，使用logloss来评估模型的表现：
$$logloss =-\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^My_{ij}\log(p_{ij})$$，
其中N表示测试集中的照片数量这里是79726，M是照片的类别数量这里是10，log是自然对数。$y_{ij}$当第i张图片属于第j类时为1，其余为0，$p_{ij}$是模型预测第i张照片属于第j类的概率。
(logarithmic loss)[http://wiki.fast.ai/index.php/Log_Loss]用来衡量分类模型的性能。一个完美的分类模型的log loss为0.
与常用的痕量标准精度(Accuracy)相比，Accuracy中计算预测的类别与实际的类别是否相同，只有是或不是两种可能。而log loss考虑的是预测的不确定性，是预测到每一个类别的概率，可以对模型有更细微的衡量。
在提交的数据没有要求某张图片的所有类别的概率和为1，因为每个概率会除以所有类别的概率和，做归一化处理。
为了避免出现log(0)的情况，对于模型预测的结果使用$max(min(p,1-10^{-15}),10^{-15})$来代替直接输出。
### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.
1、导入数据集：项目提供的训练数据集中已经将每一类的图片放在了一个单独的文件夹下，并以该类别命名，使用scikit-learn库中的load_files函数，可以很方便的得到训练数据和对应的类别标记。
2、数据预处理：本项目采用Keras来搭建深度学习网络模型，使用Tensorlow作为后端。因为
> 1. Keras的输入要求是一个4D的tensor即(nb_samples, rows, columns, channels)，因此将训练数据转化为这种格式的张量。
> 2. 预训练模型的输入都进行了额外的归一化过程，因此这里也要对这些张量进行归一化，即对所有的像素都减去
2、搭建模型：利用keras中利用“imagenet”预训练好的CNN模型(比如VGG16，ResNet50，Inception和Xception)

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
