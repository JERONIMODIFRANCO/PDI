# PYTHON COURSE
## Computer vision in a nutshell[](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/Unit_2.html?AWSAccessKeyId=AKIAJLU2ZOTUFJRMDOAA&Signature=Ylk0XD%2FfjHhbIhyJNUYRJcNdNAA%3D&Expires=1695256979#Computer-vision-in-a-nutshell)

Computer vision, commonly abbreviated as CV, could be described as a field of study that allows a computer to analyze and have understanding of an uploaded digital image or group of images, such as videos.

The main idea of ​​CV, along with robotics and other fields of study, is to improve tasks that could be exhaustive or repetitive for humans. In recent years, there have been many improvements with the invention of complex computer vision and deep learning systems, such as the well-known convolutional neural networks. These inventions shifted the point of view to solve many problems, such as facial recognition and medical images.

### Images[](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/Unit_2.html?AWSAccessKeyId=AKIAJLU2ZOTUFJRMDOAA&Signature=Ylk0XD%2FfjHhbIhyJNUYRJcNdNAA%3D&Expires=1695256979#Images)

First of all, we need to understand what exactly an image is. Colloquially, we could describe it as a visual representation of something that, by itself, is a set of many characteristics such as color, shapes, etc. For a computer, an image could be better described as a matrix, in which every value is considered a pixel, so when you are talking about a 1080p image resolution, you are referring to a specific 1080x1920 px matrix.

### 2.1   Color

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/Color_Channels.png)

In the case of a colored image, we are talking about a three-dimensional matrix where each dimension corresponds to a specific color channel (Red, green, or blue). <mark style="background: #FF5582A6;">The dimensions of this matrix will be different for different color spaces, </mark>which we will discuss further in the course.

We can describe an image in many more complex ways, like the color construction that is mainly a result of the light over the object surface. When we have something black, it is actually the lack of light. The color formation will depend on the wavelength of the main components of white light.

If you like physics as much as I do, you will find this phenomenon interesting, where the color deformation can be seen: the stars. In many pictures of space, <mark style="background: #FF5582A6;">you can see that the rock formations that are far from us have a red color, while the closest ones have a blue color. </mark>This phenomenon was discovered by the North American astronomer Edwin Hubble in 1929. We know that space is in a state of constant expansion, so if space is deformed, the light that we receive from those <mark style="background: #FF5582A6;">stars will suffer from that expansion</mark>, too. As a consequence, the wavelength of the light will be higher and the color we perceive will have a red tone instead of a blue one.

This is an open source image from NASA. You can find it at [https://images.nasa.gov/](https://images.nasa.gov/)

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/nasa_Spiral.jpg)

  

We don't want to go much deeper on the color formations and theory of it. The main idea is so you can understand the basis of what we are going to work with for the rest of the course. It will be helpful if you want to do more profound research on this topic, which I consider really interesting.

### Going into practice![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/Unit_2.html?AWSAccessKeyId=AKIAJLU2ZOTUFJRMDOAA&Signature=Ylk0XD%2FfjHhbIhyJNUYRJcNdNAA%3D&Expires=1695256979#Going-into-practice!)

Ok, so now that you have a brief introduction about what computer vision is, and a little background on image formation, it's time to describe one of the basic tasks of CV: color filtering. Color filtering is extracting from an image the information of a specific color. Before that, we will see some basic operations with opencv so you can get acquainted with this library, and understand the code ahead.

### 2.3   Loading and writing an image

Opencv has some algorithms that permit work easily with images. <mark style="background: #FF5582A6;">The principal ones are **cv2.imread** , **cv2.imwrite**,and **cv2.imshow**</mark>. For example, here is a simple code in python that uses some of these algorithms. Let's see.
```python
#Import the Opencv Library
import cv2

#Read the image file
img = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/test_image_1.jpg')

#Display the image in a window
cv2.imshow('image', img)

#Save the image "img" in the current path 
cv2.imwrite('image.jpg', img)

#The window will close after a key press
cv2.waitKey(0)
cv2.destroyAllWindows()
```

But, if we want to use it with ROS, how do they work? Let's have a look.

#### imread ,imshow and imwrite[](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/Unit_2.html?AWSAccessKeyId=AKIAJLU2ZOTUFJRMDOAA&Signature=Ylk0XD%2FfjHhbIhyJNUYRJcNdNAA%3D&Expires=1695256979#imread-,imshow-and-imwrite)
### 2.4   Space colors

Before going into color filtering, you need to understand the concept of space colors. We are going to use it often during the course. Additionally, it will help you to experiment with different color spaces for different applications. A space color is no more than a three-dimensional model that tries to describe the human perception known as color, where the coordinates of the model will define a specific color. One of them that you may know is the RGB, where all the colors are created by mixing red, green, and blue (<mark style="background: #FF5582A6;">Python works with quite a different model of RGB, inverting the order of the colors, so the final model is BGR</mark>).

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/RGB.png)

  

Just like we said at the beginning of the course, one of the main objectives is to detect colors in images. For this specific task, we will use a color space know as <mark style="background: #FF5582A6;">HSV</mark> (Hue Saturation Value) <mark style="background: #FF5582A6;">that is a closer model to how humans perceive colors.</mark> This is a <mark style="background: #FF5582A6;">non-linear model of RGB</mark> with cylindrical coordinates.

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/HSV.png)

  

For the next exercise, we will apply a color filter to the next image. The main idea of this exercise is to pull apart each of the three colors.

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/Filtering.png)

  

<mark style="background: #FF5582A6;">We use the HSV space colors because they make it easier to define the color we want. The color components themselves are defined by the Hue channel</mark>, <mark style="background: #FF5582A6;">having the entire chromatic spectrum present, compared to the RGB where we need all three channels to define a color</mark>.

For better comprehension of this part, we can make use of the following image. It is an approximation of how the colors are defined in the hue channel.

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/hsv_colors.png)

For example, if the color I'm looking for is blue, my Hue range should be between 110 and 120, or 100 and 130 for a wider range. So the value of my lower limit should look something like `min_blue = np.array([110,Smin,Vmin])` and the higher limit `max_blue = np.array([120,Smax,Vmax])`. In the case of Saturation and value, we can say that the lower the saturation, the closer to white, and the lower the value, the closer to black, as can be seen in the image below:

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/blue_c.png)

###### _Color Filtering_[](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/Unit_2.html?AWSAccessKeyId=AKIAJLU2ZOTUFJRMDOAA&Signature=Ylk0XD%2FfjHhbIhyJNUYRJcNdNAA%3D&Expires=1695256979#Color-Filtering)

Well, if you were working with just OpenCV, you would have something similar to the next code.
```python
import cv2

#Import the numpy library which will help with some matrix operations
import numpy as np 

image = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/Filtering.png')

#I resized the image so it can be easier to work with
image = cv2.resize(image,(300,300))

#Once we read the image we need to change the color space to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#Hsv limits are defined
#here is where you define the range of the color you´re looking for
#each value of the vector corresponds to the H,S & V values respectively
min_green = np.array([50,220,220])
max_green = np.array([60,255,255])

min_red = np.array([170,220,220])
max_red = np.array([180,255,255])

min_blue = np.array([110,220,220])
max_blue = np.array([120,255,255])

#This is the actual color detection 
#Here we will create a mask that contains only the colors defined in your limits
#This mask has only one dimention, so its black and white }
mask_g = cv2.inRange(hsv, min_green, max_green)
mask_r = cv2.inRange(hsv, min_red, max_red)
mask_b = cv2.inRange(hsv, min_blue, max_blue)

#We use the mask with the original image to get the colored post-processed image
res_b = cv2.bitwise_and(image, image, mask= mask_b)
res_g = cv2.bitwise_and(image,image, mask= mask_g)
res_r = cv2.bitwise_and(image,image, mask= mask_r)

cv2.imshow('Green',res_g)
```

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/green.png)

  
```python
cv2.imshow('Red',res_r)
```

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/red.png)

  

```python
cv2.imshow('Blue',res_b)
```

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/blue.png)

  
```python
cv2.waitKey(0)
cv2.destroyAllWindows()
### 2.5   Edge Detection

Edge detection in the image processing world is very important because it facilitates object recognition, region segmentation of images, and many other tasks. <mark style="background: #FF5582A6;">The edges are places on the image where an abrupt change in the levels of gray exist.</mark>

For this next chapter, we are going to work with edge detection. First, we will talk about the <mark style="background: #FF5582A6;">canny detector, which uses convolution masks and is based on the first derivative</mark>. Second is the sobel operator, which also works with convolutions. (It should be noted that the <mark style="background: #FF5582A6;">canny detector uses the sobel operator to get the first derivative in the horizontal and vertical direction for the gradient</mark>.)

##### Sobel Operator[](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/Unit_2.html?AWSAccessKeyId=AKIAJLU2ZOTUFJRMDOAA&Signature=Ylk0XD%2FfjHhbIhyJNUYRJcNdNAA%3D&Expires=1695256979#Sobel-Operator)

<mark style="background: #FF5582A6;">The [Sobel operator].</mark>(https://www.researchgate.net/publication/239398674_An_Isotropic_3x3_Image_Gradient_Operator) is used in image processing, especially in edge detection algorithms. The operator calculates the intensity [gradient](http://www.cs.umd.edu/~djacobs/CMSC426/ImageGradients.pdf) of an image in every píxel using the convolution function, and the result shows the intensity magnitude changes that could be considered edges (The convolution is a mathematical operation that can be widely used in the signal processing as a filter - it transforms two functions into a third one, representing how much it changes the second function with respect to the first one.)

##### The convolution[](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/Unit_2.html?AWSAccessKeyId=AKIAJLU2ZOTUFJRMDOAA&Signature=Ylk0XD%2FfjHhbIhyJNUYRJcNdNAA%3D&Expires=1695256979#The-convolution)

Let's suppose we have a Matrix M with **_m_** rows by **_n_** columns:

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/Mmn.png)

  

And a second matrix (this should be a square Matrix) that we will call the kernel, with **_i_** rows and **_j_** columns:

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/kij.png)

  

So<mark style="background: #FF5582A6;"> the convolution between the **_M_** matrix and the kernel **_k_** will be</mark>:

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/res.png)

  

Which <mark style="background: #FF5582A6;">results in a third matrix **_R_**:</mark>

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/Rab.png)

  

<mark style="background: #FF5582A6;">Where the rows and columns will be defined b</mark>y:

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/size.png)

  

So the components of R will be given by:

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/Sum.png)

  

For a better understanding of this operation, let's work with a **_5x5_** Matrix and a **_3x3_** kernel:

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/Convolutional.png)

  

The Convolution operation will be something like:

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/convolution.gif)

  

As can be seen in the gif above, the kernel travels around the image from left to right and top to bottom.<mark style="background: #FF5582A6;"> The jump distance of the kernel is called stride, which is commonly set as 1px.</mark>

It is important that you understand that the convolution operation, the gradient calculated by the sobel operator in an image, is made by convolving the "Sobel Filters" all over the image. <mark style="background: #FF5582A6;">We have 2 kernels, for vertical and horizontal gradients</mark>:

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/sobel_operators.png)

As always, you will see now how the codes will be if they were using just Open CV and if it works in order to use the Sobel Operator.

###### _sobelA.py_[](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/Unit_2.html?AWSAccessKeyId=AKIAJLU2ZOTUFJRMDOAA&Signature=Ylk0XD%2FfjHhbIhyJNUYRJcNdNAA%3D&Expires=1695256979#sobelA.py)
```python
import cv2
import numpy as np

img = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/test_img_b.jpg')

#Convert the image to gray scale so the gradient is better visible
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(450,350))

#Apply the horizontal sobel operator with a kernel size of 3
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)

#Apply the vertical sobel operator with a kernel size of 3
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

cv2.imshow('Original',img)

```
This is an open source image from NASA. at [https://images.nasa.gov/](https://images.nasa.gov/)

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/test_img_b.jpg)

  

```python
cv2.imshow('sobelx',sobelx)
```

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/sobely_a.jpg)

  
```python
cv2.imshow('sobely',sobely)
```

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/sobelx_a.jpg)

  
```python
cv2.waitKey(1)
cv2.destroyAllWindows()

#### Canny edge detection[](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/Unit_2.html?AWSAccessKeyId=AKIAJLU2ZOTUFJRMDOAA&Signature=Ylk0XD%2FfjHhbIhyJNUYRJcNdNAA%3D&Expires=1695256979#Canny-edge-detection)

_For a better understanding of the canny edge detector, you can visit the [Opencv Page](https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html)_

<mark style="background: #FF5582A6;">This is a multi-stage algorithm based on the sobel operator, described above. The first stage of the canny detector is Noise reduction, which is done by applying a Gaussian Filter</mark>, defined by:

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/GaussFilter.png)

<mark style="background: #FF5582A6;">where each value of the kernel is described by:</mark>

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/GaussDefinition.png)

<mark style="background: #FF5582A6;">This step will smooth the image, like the algorithm is based on the derivative to get the gradient (Sobel). This is very sensitive to noise, so this step helps to decrease that sensitivity.</mark>

<mark style="background: #FF5582A6;">The second stage is acquiring the gradients, where the intensity (Magnitude) and orientation of the edges is calculated. The first part is done by obtaining the derivatives **_Ix_** and **_Iy_**, which can be implemented by convolving the image with the horizontal and vertical Sobel kernels.</mark>

Once we have both derivatives, <mark style="background: #FF5582A6;">the magnitude **_G_** and orientation **_θ_** are defined by</mark>:

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/magnitud_Orientation.png)

<mark style="background: #FF5582A6;">After obtaining this information, a stage of [Non Maximal supression]</mark>i(https://arxiv.org/pdf/1705.02950.pdf) is applied, to eliminate pixels that could not be wanted. For this, every pixel is checked to see if it is a local maximum in its neighborhood in the direction of gradient. Finally, a stage of Hysteresis thresholding is applied, which decides which edges are really edges and which are not.

_A nice explanation of the Hysteresis can be found [here](https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123)._

As always, we show you how the code would look if you are just working with Open CV. So, see what is going to happen with the image as an example:
```python
import cv2
import numpy as np 

img = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/test_img.png')
img = cv2.resize(img,(450,350))

#The canny detector uses two parameters appart from the image:
#The minimum and maximum intensity gradient
minV = 30
maxV = 100

edges = cv2.Canny(img,minV,maxV)
cv2.imshow('Original',img)
```
This is an open source image from NASA. You can find it at [https://images.nasa.gov/](https://images.nasa.gov/).

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/test_img.png)

  

```python
cv2.imshow('Edges',edges)
```

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/canny.jpg)

  

```python
cv2.waitKey(0)

cv2.destroyAllWindows()
```
<mark style="background: #FF5582A6;">The `minV` and `maxV` are considered the limits of intensity gradient. This means that if the gradient intesity is lower than minV, this part of the image is considered a non-edge, so it will be discarded. If the value is higher than maxV, they are considered borders. Finally, those who are in between the limits will be considered edges or non-edges, depending on their connectivity</mark>.
### 2.6   Morphological Transformations

Morphological transformations are, in my personal opinion, one of the most important operations in image processing, and they can be helpful with noise supression in images and other tasks. These are simple operations based on the image form<mark style="background: #FF5582A6;"> commonly applied over a binary image</mark>. <mark style="background: #FF5582A6;">This works with a matrix kernel</mark> that can be, for example, a 5x5 matrix of ones.

<mark style="background: #FF5582A6;">Four of the most common morphological transformations are</mark>:

- <mark style="background: #FF5582A6;">Erosion</mark>
- <mark style="background: #FF5582A6;">Dilation</mark>
- <mark style="background: #FF5582A6;">Opening</mark>
- <mark style="background: #FF5582A6;">Closing</mark>

_More information about morphological transformations can be found at the [Opencv Page](https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html)._

This is the last part of this unit, but not the least important! We have the Open CV code here, and how it changes the images. Let's have a look, and then try it using the architecture of ROS, and finally applying it to the **cv_image**. Go ahead! Come on! It's almost done!

###### _main.py_[](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/Unit_2.html?AWSAccessKeyId=AKIAJLU2ZOTUFJRMDOAA&Signature=Ylk0XD%2FfjHhbIhyJNUYRJcNdNAA%3D&Expires=1695256979#main.py
```python
import cv2
import numpy as np

#Read the image in grayscale 
img = cv2.imread('/home/user/catkin_ws/src/opencv_for_robotics_images/Unit_2/Course_images/world.png',0)
img = cv2.resize(img,(450,450))

#Define a kernel for the erosion 
kernel_a = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel_a,iterations = 1)

#Define a kernel for the dilation
kernel_b = np.ones((3,3),np.uint8)
dilation = cv2.dilate(img,kernel_b,iterations = 1)

#Define a kernel for the opening
kernel_c = np.ones((7,7),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_c)

#Define a kernel for the closing
kernel_d = np.ones((7,7),np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_d)

cv2.imshow('Original',img)
```

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/world.png)

  

```python
cv2.imshow('Erosion',erosion)
```

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/erosion.jpg)

<mark style="background: #FF5582A6;">The erotion transformation provokes the blacks to be wider than the original image.</mark>

```python
cv2.imshow('Dilation',dilation)
```

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/dilation.jpg)

  

<mark style="background: #FF5582A6;">Unlike the erosion, the dilation provokes the whites to be wider, as you can see as the borders of the circle became thinner.</mark>

```python
cv2.imshow('Opening',opening)
```

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/closing.jpg)

  

The opening and the closing are my favourites. They help to eliminate little dots that can be considerewd noise in the image. In the case of the opening, it takes little black dots as noise and supresses them.

```python
cv2.imshow('Closing',closing)
```

![](https://s3.eu-west-1.amazonaws.com/notebooks.ws/opencv_robotics/images/opening.jpg)

  

<mark style="background: #FF5582A6;">The closing is similar to the opening, it works with white noise. As you can see, the inner white dots in the circle were almost eliminated from the image.
</mark>
```python
cv2.waitKey(0)
cv2.destroyAllWindows()
```
You can modify the parameters of the transformations to make the effect of them stronger or weaker. It will depend on the application you want.
