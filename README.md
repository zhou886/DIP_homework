# 数字图像处理大作业

[toc]

## 问题描述

缩放是常见的图像处理操作，因为图像中每个像素的重要程度与图像内容密切相关，所以只考虑几何参数调整并非最理想的处理方式。

本项目的目标是设计一个算法，输入一张原始图片和指定的长度和宽度。然后经过算法的处理，输出一张变成指定的长度和宽度后的图像，要求考虑像素的差异化处理。

## 问题分析

### 问题背景

随着社会进入信息化时代，图像和视频这两种媒体成为表示和传递信息的优秀介质。在现实中，手机已越发成为信息传递的终端，而如何在这些小尺寸屏幕上显示图像也成为一个重要问题。

在布局方面，工程师提出了响应式布局的方式，它会在不同尺寸，不同比例的屏幕上自动调整显示内容的布局，有助于让图像能更加完整地显示。但是这种方法依然有它的局限性，受制于图像本身尺寸，无论通过什么方法都无法将一张像素点很多的图像完整地呈现出来。

### 常用方法

现有常用的图像缩放方法主要有放缩和裁切两种。前者会导致图像比例出现问题，几何关系不正常，表现效果较差；后者则因难以分辨图像中的哪些部分重要，哪些部分不重要，从而导致可能会裁切掉图像的重要部分导致其所要传达的信息不完整或损坏。

为避免以上传统方法带来的问题，有必要使用智能化图像放缩的方法对图像大小进行调整，本作业采用改进后的Seam Carving方法对图像大小进行调整。

## 实现方法

### 概述

Seam Carving方法是通过尽可能删除或者增加图像中不重要的内容，保留图像中重要的部分来实现图像大小的改变的。

<img src=".\README.assets\example.jpg" alt="example" style="zoom:50%;" />

比如上面这张图像，人观察该图像时，注意力会放在图像中间的建筑上，那么这部分内容就是图像中的重要部分，而上面的蓝天和下面的水面则都是相对不太重要的部分。

Seam Carving方法会寻找连接图片上下边缘或者左右边缘的缝(seam)，这些缝都是图像中相对不重要的部分。连接上下边缘的缝是从每行中都选择一个像素，这些像素之间为八邻域的关系；连接左右边缘的缝是从每列中选择一个像素，这些像素之间也是八邻域的关系。

该方法通过能量函数的方法来量化衡量不同像素的重要程度：重要部分的像素能量函数的值较大，不重要部分的像素能量函数的值较小。常用的一种能量函数定义如下：
$$
e_1(I) = \abs{\frac{\part }{\part x}I} + \abs{\frac{\part}{\part y}I}
$$
上述能量函数实际上就是计算每个像素的x方向梯度与y方向梯度的绝对值，再求和，也就是对该图像进行边缘提取。一般认为梯度变化剧烈的地方，也即物体的边缘，是一副图像中重要的部分，应当予以保留，比如上图中建筑物的边缘。而梯度变化不明显的地方则相对不那么重要，比如上图中的天空和水面。

### 使用Seam Carving缩小图像

使用Seam Carving方法对图像进行缩小的步骤如下：

1. 根据公式(1)计算图像的能量图。
2. 根据能量图计算代价图和路径图。
3. 根据代价图和路径图选择一条最优路径。
4. 删除该最优路径。
5. 重复1~4直到图像缩小到指定大小。

### 使用Seam Carving放大图像

使用Seam Carving方法对图像进行放大的步骤如下：

1. 根据公式(1)计算图像的能量图。
2. 根据能量图计算代价图和路径图。
3. 根据代价图选择k条最优路径，k为要增大的数量。
4. 将k条最优路径平均插入图像中。

### 计算图像的能量图

一般常见的边缘检测的方法有Roberts算子、Prewitt算子、Sobel算子和Canny边缘检测器。对上述四种方法都进行实现，并验证哪一种方法效果更好。

1. Roberts算子
   $$
   dx = \left [
   \begin{matrix}
   -1 & 0
   \\
   0 & 1
   \end{matrix}
   \right ]
   \quad
   dy = \left [
   \begin{matrix}
   0 & -1
   \\
   1 & 0
   \end{matrix}
   \right ]
   $$

2. Prewitt算子
   $$
   dx = 
   \left [ 
   \begin{matrix}
   1 & 1 & 1
   \\
   0 & 0 & 0
   \\
   -1 & -1 & -1 
   \end{matrix}
   \right ]
   
   \quad
   
   dy =
   \left [
   \begin{matrix}
   -1 & 0 & 1
   \\
   -1 & 0 & 1
   \\
   -1 & 0 & 1
   \end{matrix}
   \right ]
   $$

3. Sobel算子
   $$
   dx = 
   \left [ 
   \begin{matrix}
   -1 & 0 & 1
   \\
   -2 & 0 & 2
   \\
   -1 & 0 & 1
   \end{matrix}
   \right ]
   
   \quad
   
   dy =
   \left [
   \begin{matrix}
   -1 & -2 & -1
   \\
   0 & 0 & 0
   \\
   1 & 2 & 1
   \end{matrix}
   \right ]
   $$

### 计算代价图和路径图

连接图像上下边缘的缝可以定义为：
$$
s^x = \{s_i^x\}_{i=1}^n = \{ (x(i), i) \}_{i=1}^n, \quad s.t. \forall i \ \abs{x(i) - x(i-1)} \le 1
$$
其中，$$x(i)$$表示缝的第i行像素所在的位置，要求缝的每个相邻两行之间的像素所在位置差不能超过1，也就是每个像素都处于相邻像素的八邻域内。

同理，连接图像左右边缘的缝可以定义为：
$$
s^y = \{s_j^y\}_{j=1}^m = \{ (j, y(j)) \}_{j=1}^m, \quad s.t. \forall j \ \abs{y(j) - y(j-1)} \le 1
$$
对于连接图像上下边缘的缝中处于第i行第j列的像素，它只能连接第i-1行中位于j-1列、j列和j+1列的像素。根据动态规划的思想，若要求$$\sum_{i=1}^{n} e(i, x(i))$$最小，则需要选择上述三个像素中能量最小的一个。所以可以使用下面的递推公式来计算代价图：
$$
M(i, j) = e(i, j) + min(M(i-1, j-1),\ M(i-1, j),\ M(i-1, j+1))
$$
当代价图计算的同时，每个缝所对应的路径也被计算，即路径图。

## 效果展示

### 实验环境

|    测试环境    |       配置       |
| :------------: | :--------------: |
|    操作系统    |  Windows10 20H2  |
|      CPU       |     I7-9750H     |
|      RAM       | 16G DDR4 2133MHz |
| Python Version |   3.9.7 64bit    |
| opencv-python  |     4.5.4.60     |
|     numpy      |      1.20.1      |



### 效果

由于Seam Carving算法速度偏慢，所以我自己绘制了一个简单的图像作为测试用例，该测试图像长度与宽度均为300pix。原图如下所示。

<img src="C:\CODE\Python\DIP_homework\README.assets\test.png" alt="test" style="zoom:50%;" />

设置输出的图像长度与宽度均为200pix，测试不同边缘检测方法的效果对图像缩小的效果。

| 边缘检测方法 |                           输出图像                           |
| :----------: | :----------------------------------------------------------: |
|   roberts    | <img src="C:\CODE\Python\DIP_homework\README.assets\roberts.png" alt="roberts" style="zoom: 67%;" /> |
|   prewitt    | <img src="C:\CODE\Python\DIP_homework\README.assets\prewitt.png" alt="prewitt" style="zoom: 67%;" /> |
|    sobel     | <img src="C:\CODE\Python\DIP_homework\README.assets\sobel.png" alt="sobel" style="zoom: 67%;" /> |
|    canny     | <img src="C:\CODE\Python\DIP_homework\README.assets\canny.png" alt="canny" style="zoom: 67%;" /> |

设置输出的图像长度与宽度均为400pix，测试不同边缘检测方法的效果对图像放大的效果。

| 边缘检测方法 |                           输出图像                           |
| :----------: | :----------------------------------------------------------: |
|   roberts    | <img src="C:\CODE\Python\DIP_homework\README.assets\reberts2.png" alt="reberts2" style="zoom:50%;" /> |
|   prewitt    | <img src="C:\CODE\Python\DIP_homework\README.assets\prewitt2.png" alt="prewitt2" style="zoom:50%;" /> |
|    sobel     | <img src="C:\CODE\Python\DIP_homework\README.assets\sobel2.png" alt="sobel2" style="zoom:50%;" /> |
|    canny     | <img src="C:\CODE\Python\DIP_homework\README.assets\canny2.png" alt="canny2" style="zoom:50%;" /> |

可见，Roberts算子和Sobel算子的效果要显著好于Prewitt算子和Canny边缘检测器。

## 不足、原因及其改进

### 不足

经过大量测试，我认为原有的Seam Carving方法存在以下问题：

1. 相比于矢量图，Seam Carving在位图，尤其是自然风景类的图像中效果更好，无论使用哪一种边缘检测方法。
2. Prewitt算子和Canny边缘检测器效果比较差，不符合预期，当时预计Canny边缘检测器效果应该较好。
3. 该方法不能将图像放大到一倍以上的大小，放大比例受到限制。
4. 图像部分几何关系被破坏，边缘的平行关系消失或被破坏，部分直线弯曲，物体边缘过渡不自然。

### 原因

经过思考，我认为上述问题分别对应一下的原因：

1. 位图中，人眼对很多信息关注少，类比于JPEG图像压缩技术，更容易将一些不显眼的缝删除或者增加。而在矢量图中，图像各区域之间边界清晰，任何一点对边界的破坏都会导致图像看起来不自然。

2. 对比四种边缘检测方式得到的能量图，可以发现：

   | 检测方式 |                            能量图                            |
   | :------: | :----------------------------------------------------------: |
   | roberts  | <img src="C:\CODE\Python\DIP_homework\README.assets\roberts_energy_map.png" alt="roberts_energy_map" style="zoom:67%;" /> |
   | prewitt  | <img src="C:\CODE\Python\DIP_homework\README.assets\prewitt_energy_map.png" alt="prewitt_energy_map" style="zoom:67%;" /> |
   |  sobel   | <img src="C:\CODE\Python\DIP_homework\README.assets\sobel_energy_map.png" alt="sobel_energy_map" style="zoom:67%;" /> |
   |  canny   | <img src="C:\CODE\Python\DIP_homework\README.assets\canny_energy_map.png" alt="canny_energy_map" style="zoom:67%;" /> |

   Roberts和Sobel算子都只对横向的边缘检测效果比较好，而对于纵向的边缘检测效果却很差，而Prewitt算子和Canny边缘检测器则对两个方向的边缘都检测的比较好。问题就出在这里，Seam Carving方法选择路径时是在一个像素的八邻域中选择的，所以可以斜着选择像素，Canny边缘检测器对于不连续的线段间不会将其断开处连接起来，Prewitt算子虽然在断开处有值，但相比周围的边缘都偏小。于是根据算法，它会选择能量小的断开处的像素，于是就有了多次穿越边缘的缝。对于这些缝的操作导致了图像主要内容的畸变。

3. 对于不能将图像放大一倍以上的问题原因在于算法中是一次性选择k个最优的缝，然后将其插入到原图像中。因为是一次操作，所以k不能超过原图像的要操作的那边的宽度，否则就没有缝可以选择了。

4. 图像部分几何关系被破坏的原因在于多条删去或增加的缝经过了边缘。受限于图像本身的限制，缝几乎都必须穿过一些边缘，但不应该同时出现多条缝穿过同一区域的边缘。

### 改进

经过思考、收集资料和测试，我对上述问题分别提出以下改进方法：

1. 对矢量图采取特别的措施，特别使用改进递推公式后的canny检测器作为能量函数。

2. 改进计算图的递推公式，使用约束项来减少缝穿过断开处的次数。改进后的递推公式如下所示：
   $$
   M(i,j) = e(i,j) + min(M(i-1,j-1) + C_L(i,j), M(i-1,j) + C_U(i,j), M(i-1,j+1)+C_R(i,j))
   \\ \\
   其中\left \{
   \begin{array}\left
   C_L(i,j) = \abs{I(i-1,j) - I(i,j-1)} + \abs{I(i,j-1) - I(i,j+1)}
   \\
   C_U(i,j) = \abs{I(i,j-1) - I(i,j+1)}
   \\
   C_R(i,j) = \abs{I(i,j-1) - I(i,j+1)} + \abs{I(i,j+1) - I(i-1,j)}
   \end{array}
   \right .
   $$

3. 分批多次对图像进行放大，而不是在一次内完成。

4. 对已穿过的边缘进行加强处理，避免之后其他缝再次穿过该条边缘。

