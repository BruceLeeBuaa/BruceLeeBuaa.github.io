## Tensorflow基础知识

###### 数据类型

int，float，double，bool，string

```python
# tensor 类型
tf.constant(1)						# 创建一个tensor, int型
tf.constant(1.0)					# 创建一个tensor, float型
tf.constant(1.0,dtype=tf.double)	# 创建一个tensor, double型

# tensor 属性
with tf.device("cpu"):
    a = tf.constant([1])
a.device	# 'cpu'
a.gpu()		# 把a迁移到gpu上
a.numpy()	# 把a变为numpy变量
a.shape		# a的大小
a.ndim		# a的维度
tf.is_tensor(a) # 看a是否为tensor，True of False
a.dtype		# a的数据类型

# tensor 类型转换
a = np.arrange(5) 			   # array([0,1,2,3,4]), 默认为int64类型
aa = tf.convert_to_tensor(a,dtype=tf.int32) # 转换为int32的tensor
tf.cast(aa,dtype=tf.float32)   # array([0.,1.,2.,3.,4.])
# 整型与bool型的转换
b = tf.constant([0,1]) 
bb = tf.cast(b, dtype=tf.bool) # array([False, True])

# Variable: 可求导
b = tf.Variable(a)
b.trainable             	   # True
a.numpy() 					   # 把a转换为numpy类型
float(a)					   # 把a转换为float类型
```

###### 创建tensor

tf.constant,  tf..convert_to_tensor : 参数是具体数值

tf.zeros, tf.fill, tf.ones : 参数是size大小

随机初始化: 参数是size大小

```python
tf.convert_to_tensor(np.ones([2,3]))
tf.convert_to_tensor([1.,2.])			# 初始化数据为1，2的tensor
tf.zeros([m,n])							# shape : m * n
tf.zeros(a.shape) == tf.zeros_like(a)   # tf.ones() 同理
tf.fill([m,n],p)						# shape : m*n, 元素全为p
# 随机初始化
tf.random.normal([2,2],mean=1,stddev=1) 		  # 正态分布，N(1,1)
tf.random.truncated_normal([2,2],mean=0,stddev=1) # 被裁剪的正态分布
tf.random.uniform([2,2],minval=0,maxval=1)        # 均匀分布，U(0,1)
# 其他功能
idx = tf.range(10)						# [0,1,2,...]
idx = tf.random.shuffle(idx)			# [2,1,9,...],随机打散
a = tf.random.normal([10,784])
b = tf.random.normal([10])
a = tf.gather(a,idx),b = tf.gather(b,idx) # 在打散的同时不破坏a,b的对应关系

# 不同维度tensor
# scalar 标量
loss = tf.keras.losses.mse(y,out)		# mse loss
loss = tf.reduce_mean(loss)				# 对每一个mse loss再求平均
# 一维 vector
net = layers.Dense(10)
# 二维 matrix
x = tf.random.normal([4,784])
net = layers.Dense(10)
net.build((4,784))
net(x).shape						# [4,10]=[4,784]*[784,10]+[10]
net.kernel.shape					# [784,10]
net.bias.shape						# [10]
# 三维tensor:embedding ；四维tensor:图片
```

###### 索引和切片

许多操作和pytorch类似，这里就不过多赘述了

```python
# numpy 索引方式
a = tf.random.normal([4,3,28,28])
a[1,2,3,2].shape					# TensorShape([]) 标量

# 切片
a[:2].shape  			   # [2,3,28,28] 左侧包含右侧不包含
a[:2,-1:,:,:].shape 	   # [2,1,28,28]
a[:,:,0:28:2,0:28:2].shape # [4,3,14,14] 第二个冒号后代表步长
a[:,:,::2,::2].shape	   # [4,3,14,14] 和上面等价
a[0,...].shape			   # [1,3,28,28] 省略号代表后面都取

# selective_index          a:[4,35,8]
tf.gather(a,axis=0,indices=[2,3]).shape		# [2,35,8]
tf.gather(a,axis=0,indices=[2,1,3,0]).shape # [4,35,8], 相当于重排序

tf.gather_nd(a,[0]).shape       	 # [35,8]
tf.gather_nd(a,[0,1]).shape     	 # [8]
tf.gather_nd(a,[0,1,2]).shape   	 # []
tf.gather_nd(a,[[0,1,2]]).shape    	 # [1],即[标量]
tf.gather_nd(a,[[0,0],[1,1]]).shape  # [2,8],8+8=2*8

# a : [4,28,28,3]
tf.boolean_mask(a,mask=[True,True,False],axis=3).shape # [4,28,28,2]
```

###### 维度变换

```python
# size变换
a = tf.random.normal([4,28,28,3])	 # [b,h,w,c]
a.shape a.ndim						 # [4,28,28,3], 4
tf.reshape(a,[4,-1,3])				 # [4,784,3]
tf.reshape(tf.reshape(a,[4,-1]),[4,14,56,3])	# [4,14,56,3]
# 维度重排
a = tf.random.normal([4,3,2,1])
tf.transpose(a).shape        		 # [1,2,3,4]
tf.transpose(a,perm=[0,1,3,2]).shape # [4,3,1,2]
# 对比pytorch [b,c,h,w]   而tensorflow是 [b,h,w,c] 默认content格式
tf.transpose(a,perm=[0,3,1,2]).shape # 把tensorflow数据转为pytorch
# 维度扩张
a = tf.random.normal([4,35,8])
tf.expand_dims(a,axis=3).shape       # [4,35,8,1],插入输入的维度
tf.expand_dims(a,axis=-1).shape      # [4,35,8,1]
tf.expand_dims(a,axis=-4).shape		 # [1,4,35,8]

tf.squeeze(tf.zeros([1,2,1,1,3])).shape    # [2,3]
a = tf.zeros([1,2,1,3])					   
tf.squeeze(a,axis=2).shape				   # [1,2,3],删去第2个维度	
```

###### broadingcast

feature map : [4, 32, 14, 14] bias : [32,1,1] — [1,32,1,1] — [4,32,14,14]

与pytorch类似，broadcast那一维必须为1

```python
# 显式的broadcast
b = tf.broadcast_to(tf.random.normal([4,1,1,1]),[4,32,32,3])
```

###### 数学运算

* 加减乘除(element-wise):a+b,a-b,a*b,a/b,a//b,a%b
* 矩阵乘法

```python
tf.math.log(a)					# 对a所有元素以e为底取对数
tf.exp(a)						# 对a所有元素以e为底取指数
# 如果想实现以2为底的对数用换底公式，log23=loge3/loge2
tf.pow(b,3)						# 幂
tf.sqrt(b)						# 开方
# 矩阵乘法
tf.matmul(a,b)
```

###### 前向传播

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
# x:[60k,28,28]
# y:[60k]
(x,y), _ = datasets.mnist.load_data()		# 加载数据集
# x: [0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)

# [b, 784] => [b, 256] => [b, 128] => [b, 10]
# w:[dim_in, dim_out], b:[dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

for step, (x, y) in enumerate(train_db): # for every batch
        # x: [128, 28, 28]
        # y: [128]
        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28*28])
        with tf.GradientTape() as tape: # tf.Variable
            # [b, 784] => [b,10]
            h1 = x@w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2@w3 + b3

            # y: [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)

            # mse = mean(sum(y-out)^2),之后转换为scalar
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)

        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # w1 = w1 - lr * w1_grad 原地更新
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
```

###### 合并与分割

* tf.concat:拼接, 在某一维度拼接，其余维度的size必须相同

* tf.stack: 拼接，新增对应输入维度，该维度取0，1分别对应拼接的两个张量
* tf.split:

```python
a = tf.ones([4,35,8])
b = tf.ones([2,35,8])
c = tf.concat([a,b],axis=0)
c.shape									# [6,35,8]
d = tf.ones([2,35,8])
tf.stack([b,d],axis=0).shape			# [2,2,35,8]
# e:[2,4,35,8]
res = tf.split(e,axis=3,num_or_size_splits=2)	
len(res) # 2
res[0].shape # [2,4,35,4]
res = tf.split(e,axis=3,num_or_size_splits=[2,2,4])	# 3份，长度为2,2,4
```

###### 数据统计

* tf.norm: 张量范数，二范数
* tf.reduce_max/min/mean：求最大最小值, 均值
* tf.argmax/argmin：最大最小值的位置
* tf.equal: 判断是否相等
* tf.unique: 去掉重复元素

```python
b = tf.ones([2,2])
tf.norm(b,ord=2,axis=1)					# 在维度1上求二范数
a = tf.random.normal([4,10])			
tf.reduce_min(a,axis=1)					# 求维度1上的最大值

a = tf.constant([1,2,3,2,5])			
b = tf.range(5)							# [0,1,2,3,4]
tf.equal(a,b)							# [false,.....]
tf.reduce_sum(tf.cast(tf.equal(a,b),dtype=tf.int32)) # 0,~准确率
```

###### 张量排序

* sort/argsort：排序 / 排序后的每个元素原来出现的位置
* top_k: 返回最大的前k个元素
* top_k准确率

```python
a = tf.random.shuffle(tf.range(5))			# [2,0,3,4,1]
tf.sort(a,direction='DESCENDING')			# 降序排列, [4,3,2,1,0]
idx = tf.argsort(a,direction='DESCENDING')	# [3,2,0,4,1]
tf.gather(a,idx)							# 相当于还原, [4,3,2,1,0]

tf.math.top_k(a,2)							# 返回a的前两个元素

# top_k accuracy
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred, perm=[1, 0])
    target_ = tf.broadcast_to(target, pred.shape)
    # [10, b]
    correct = tf.equal(pred, target_)
    
    res = []
    for k in topk:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k * (100.0 / batch_size))
        res.append(acc)
    return res
```

###### 数据填充与复制

* pad：在原张量的每一维度的左右填充新的行
* tile：张量复制

```python
tf.pad(x,[[a,b],[c,d]])				# 假设x是一个3*3的张量矩阵，在行维度上，矩阵上下分别填充a行和b行。之后，在列的维度上，前后分别填充c行和d行，默认填充0
a = tf.random.normal([4,28,28,3])
b = tf.pad(a,[[0,0],[2,2],[2,2],[0,0]])	# [4,32,32,3]

a 									# 3*3的tensor
tf.tile(a,[1,2])					# 3*6, 因为第一个维度是1代表不复制，第二个维度是2，代表第二个维度复制两遍
```

###### 张量限幅

* clip_by_value
* relu
* clip_by_norm：等比例放缩，张量方向不变，只改变大小
* clip_by_norm：防止梯度爆炸

```python
tf.clip_by_value(a,2,8)				# 将a种元素限制在2~8之间
tf.maximum(a,0) = tf.nn.relu(a)

a = tf.random.normal([2,2],mean=10)
tf.clip_by_norm(a,15)				# a的norm变为15

grads, _ = tf.clip_by_global_norm(grads,15)	# clip_by_norm
```

###### 高阶操作

* where
* scatter_nd：三个参数：indices,updates,shape
* meshgrid：生成坐标系

```python
# where(condition,A,B)			满足条件的位置取a，反之取b,与pytorch类似
# where+单一参数				 假设a是一个3*3的张量
mask = a > 0					# >0的位置变为true, 反之则为false
tf.boolean_mask(a,mask)			# 去掉false的元素，得到一个一维array

indices = tf.where(mask)		# 返回一个array,每个元素都是true的一个坐标
tf.gather_nd(a,indices)			# 去掉false的元素，得到一个一维array

# scatter_nd
indices = tf.constant([[4],[3],[1],[7]]) # 位置
updates = tf.constant([9,10,11,12])		 # 数
shape = tf.constant([8])
tf.scatter_nd(indices,updates,shape)	 # [0,11,0,10,9,0,0,12]

# meshgrid
x = tf.linspace(-2,2,5)					 # -2~2 取 5 个点 
points_x, points_y = tf.meshgrid(x,y)
points = tf.stack([points_x, points_y],axis=2)
```