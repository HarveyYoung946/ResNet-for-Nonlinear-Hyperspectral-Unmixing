from  tensorflow.keras import layers,constraints,Sequential,initializers,regularizers,Model
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops
from  tensorflow import dtypes
from libtiff import TIFF
import VCA
import matplotlib.pyplot as plt
import scipy.io as scio

def filter_data(data,kernel_size=3,strides=3):

   if isinstance(data,list):
       data = np.array(data)
       if data.shape[-1] == 1:
           data = np.squeeze(data,axis=-1)
       if len(data.shape) != 2:
           data = np.expand_dims(data,axis=0)

   if data.shape[1] % strides == 0:
       #filter max data
       #filter min data
       print(data.shape)
       interg = int(data.shape[1] / strides)
       filter_result = np.zeros([data.shape[0],interg*2])
       t = 0
       for i in range(0,data.shape[1],strides):
            data_max = np.max(data[:, i: i + kernel_size],1)
            data_min = np.min(data[:, i: i + kernel_size],1)
            index_max = np.argmax(data[:, i: i + kernel_size], 1)
            index_min = np.argmin(data[:, i: i + kernel_size], 1)
            #记录大数在前，小数在后
            index_max_min = index_min > index_max
            #记录所有大数在前的index
            data_index_max_min = np.argwhere(index_max_min == 1)
            data_index_min_max = np.argwhere(index_max_min == 0)
            #大数在前，索引集合
            filter_result[data_index_max_min, t] = data_max[data_index_max_min]
            filter_result[data_index_min_max, t] = data_min[data_index_min_max]
            #小数在前，索引集合
            filter_result[data_index_min_max, t+1] = data_max[data_index_min_max]
            filter_result[data_index_max_min, t+1] = data_min[data_index_max_min]
            t = t + 2
   else:
       interg = int(data.shape[1]/strides)
       mod = data.shape[1] % strides
       filter_result = np.zeros([data.shape[0], (interg+1)*2])
       # filter max data
       # filter min data
       t = 0
       for i in range(0, data.shape[1]-mod, strides):
           data_max = np.max(data[:, i: i + kernel_size], 1)
           data_min = np.min(data[:, i: i + kernel_size], 1)
           index_max = np.argmax(data[:, i: i + kernel_size], 1)
           index_min = np.argmin(data[:, i: i + kernel_size], 1)
           # 记录大数在前，小数在后
           index_max_min = index_min > index_max
           # 记录所有大数在前的index
           data_index_max_min = np.argwhere(index_max_min == 1)
           data_index_min_max = np.argwhere(index_max_min == 0)
           # 大数在前，索引集合
           filter_result[data_index_max_min, t] = data_max[data_index_max_min]
           filter_result[data_index_min_max, t] = data_min[data_index_min_max]
           # 小数在前，索引集合
           filter_result[data_index_min_max, t + 1] = data_max[data_index_min_max]
           filter_result[data_index_max_min, t + 1] = data_min[data_index_max_min]
           t = t + 2
       data_max = np.max(data[:,data.shape[1]-mod:data.shape[1]],1)
       data_min = np.min(data[:,data.shape[1]-mod:data.shape[1]],1)
       index_max = np.argmax(data[:,data.shape[1]-mod:data.shape[1]],1)
       index_min = np.argmin(data[:,data.shape[1]-mod:data.shape[1]],1)
       # 记录大数在前，小数在后
       index_max_min = index_min > index_max
       # 记录所有大数在前的index
       data_index_max_min = np.argwhere(index_max_min == 1)
       data_index_min_max = np.argwhere(index_max_min == 0)
       # 大数在前，索引集合
       filter_result[data_index_max_min, t] = data_max[data_index_max_min]
       filter_result[data_index_min_max, t] = data_min[data_index_min_max]
       # 小数在前，索引集合
       filter_result[data_index_min_max, t + 1] = data_max[data_index_min_max]
       filter_result[data_index_max_min, t + 1] = data_min[data_index_max_min]

   return filter_result

def LoadData(path_data):
    data = np.loadtxt(path_data)
    data = np.array(data)
    #data = np.transpose(data)
    #data = filter_data(data, kernel_size=5, strides=5)
    return data
    #return data

class My_Initializer(initializers.Initializer):
    def __init__(self, mean,dtype=dtypes.float32):
            self.dtype = dtypes.as_dtype(dtype)
            self.mean = mean
    def __call__(self,shape,dtype=None,**kwargs):
        #data_zero = tf.zeros(shape,dtype=dtype)
        if self.dtype is None:
            dtype = self.dtype
        path_data = 'Cuprite_ori_std.txt'
        data_filtered = LoadData(path_data)
        print(data_filtered.shape)
        initial_data = tf.convert_to_tensor(data_filtered,dtype=dtype)
        print(initial_data.shape)
        return initial_data

    def get_config(self):
        # We don't include `verify_shape` for compatibility with Keras.
        # `verify_shape` should be passed as an argument to `__call__` rather
        # than as a constructor argument: conceptually it isn't a property
        # of the initializer.
        return {"dtype": self.dtype.name,"mean": self.mean}

class VCA_Initializer(initializers.Initializer):

    def __init__(self, data,num,dtype=dtypes.float32):
            self.dtype = dtypes.as_dtype(dtype)
            self.num = num
            self.data = data

    def __call__(self,shape,dtype=None,**kwargs):
        #data_zero = tf.zeros(shape,dtype=dtype)
        if self.dtype is None:
            dtype = self.dtype
        #np.random.seed(123)
        np.random.seed(1234)
        #np.random.seed(12345)
        #np.random.seed(123456)
        #np.random.seed(1234569)
        img = self.data
        data_stack = np.transpose(img.reshape(img.shape[0], img.shape[1]))
        Ae, indice, Yp = VCA.vca(data_stack, self.num)
        wavelength = np.linspace(1, data_stack.shape[0], data_stack.shape[0])
        for i in range(0, self.num):
            plt.plot(wavelength, Ae[:, i])
        plt.show()
        initial_data = tf.convert_to_tensor(np.transpose(Ae), dtype=dtype)
        return initial_data

    def get_config(self):
        # We don't include `verify_shape` for compatibility with Keras.
        # `verify_shape` should be passed as an argument to `__call__` rather
        # than as a constructor argument: conceptually it isn't a property
        # of the initializer.
        return {"dtype": self.dtype.name,"num": self.num}

class BMM_gamma_constraint(constraints.Constraint):

    def __init__(self,min_value,max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self,w,*args, **kwargs):

        w = K.clip(w, self.min_value, self.max_value)

        return w
    def get_config(self):
        return {
                'min_value':self.min_value,
                'max_value':self.max_value,
                }

class bias_contraints_MinMax(constraints.Constraint):

    def __init__(self,min_value,max_value):

        self.min_value = min_value
        self.max_value = max_value


    def __call__(self, w,*args, **kwargs):
        w = K.clip(w, self.min_value, self.max_value)

        return w

    def get_config(self):
        return {'min_value':self.min_value,
                'max_value':self.max_value}


class Normalize(layers.Layer):

    def __init__(self,min_value,**kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.min_value = min_value

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def l1_regulization(self,vector):

        return regularizers.l1(1)(vector)

    def call(self,vectors,axis=-1):

        vector = tf.nn.softmax(vectors)
        self.add_loss(self.l1_regulization(vector))
        vector = vector * math_ops.cast(math_ops.greater_equal(vector, self.min_value), K.floatx())
        sum_norm = tf.reduce_sum(vector, axis=axis, keepdims=True)
        scale = 1./ (sum_norm + K.epsilon())
        vector = tf.multiply(scale,vector)
        return vector

class BottleNeck(layers.Layer):
    def __init__(self,input_channel,base_channel,kernel_init,**kwargs):
        super(BottleNeck, self).__init__(**kwargs)

        scale = input_channel // base_channel
        self.scale = scale
        self.dense_block = []
        kernel_size = [3,5,7,9]
        for i in range(scale):
            self.dense_block.append(self.build_basic_block(base_channel=base_channel,kernel_init=kernel_init,kernel_size=kernel_size[i]))

        self.bn = layers.BatchNormalization()
        self.conv = layers.Conv1D(filters=input_channel, kernel_size=1, strides=1, padding='same',
                                  kernel_initializer=kernel_init)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def call(self, inputs,**kwargs):

        res_block = []
        for i in range(self.scale):
            x = self.dense_block[i](inputs)
            res_block.append(x)
        if self.scale != 1:
            x = layers.concatenate(res_block,axis=-1)
        x = self.bn(x)
        x = self.conv(x)
        return x

    def build_basic_block(self,base_channel,kernel_init,kernel_size):

        block = Sequential()
        # block.add(layers.BatchNormalization())
        # block.add(layers.Conv1D(filters=4*base_channel, kernel_size=1, strides=1, padding='same',kernel_initializer=kernel_init))
        block.add(layers.BatchNormalization())
        block.add(layers.Conv1D(filters=base_channel, kernel_size=kernel_size, strides=1, padding='same',kernel_initializer=kernel_init))

        return block

class ResDense_block(layers.Layer):
    def __init__(self,num_block,input_channel,base_channel,kernel_init):
        super(ResDense_block, self).__init__()
        self.bbnk = self.build_bottle_neck(num_block=num_block,input_channel=input_channel,base_channel=base_channel,kernel_init=kernel_init)
        self.bn = layers.BatchNormalization()
        self.conv = layers.Conv1D(filters=input_channel,kernel_size=1,strides=1,padding='same',kernel_initializer=kernel_init)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def call(self, inputs, **kwargs):

        x = inputs
        x = self.bbnk(x)
        #x = self.bn(x)
        x = self.conv(x)
        x = layers.add([inputs,x])
        return x

    def build_bottle_neck(self,num_block,input_channel,base_channel,kernel_init):

        res_block = Sequential()
        res_block.add(BottleNeck(input_channel=input_channel,base_channel=base_channel,kernel_init=kernel_init))
        for _ in range(1,num_block):
            res_block.add(BottleNeck(input_channel=input_channel, base_channel=base_channel, kernel_init=kernel_init))
        return res_block

class Res_Next_DenseNet(layers.Layer):
    def __init__(self,num_layer,num_block,input_channel,base_channel,kernel_init):
        super(Res_Next_DenseNet, self).__init__()

        self.num_layer = num_layer
        self.brbk = []
        for _ in range(num_layer):
            self.brbk.append(ResDense_block(num_block=num_block,input_channel=input_channel,base_channel=base_channel,kernel_init=kernel_init))

        self.bn = layers.BatchNormalization()
        self.conv = layers.Conv1D(filters=input_channel, kernel_size=1, strides=1, padding='same',
                                  kernel_initializer=kernel_init)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def call(self, inputs, **kwargs):
        x = inputs
        res_block = []
        for i in range(self.num_layer):
            x = self.brbk[i](x)
            res_block.append(x)
        if self.num_layer != 1 :
            x = layers.concatenate(res_block,axis=-1)
        #x = self.bn(x)
        x = self.conv(x)
        #x = layers.add([inputs,x])
        return x

class Attention_Layer(layers.Layer):
    def __init__(self):
        super(Attention_Layer, self).__init__()
        self.dense_1 = layers.Dense(1,activation='softmax',use_bias=False)

    def call(self, inputs, **kwargs):
        inputs_transpose = tf.transpose(inputs,perm=[0,2,1])
        a = self.dense_1(inputs_transpose)
        #a = tf.nn.softmax(a,axis=1)
        return tf.reduce_sum(inputs_transpose * a, axis=1)


class Extract_Features_Refs(layers.Layer):
    def __init__(self, filter_num, kernel_init=initializers.he_normal(), **kwargs):
        super(Extract_Features_Refs, self).__init__(**kwargs)
        self.conv = layers.Conv1D(filters=filter_num,kernel_size=30,strides=15,padding='same',kernel_initializer=kernel_init)
        self.net = ResDense_block(num_block=2,input_channel=filter_num,base_channel=filter_num//4,kernel_init=kernel_init)
        self.attention = Attention_Layer()

    def get_config(self):

        cfg = super().get_config()

        return cfg

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv(x)
        x = self.net(x)
        x = self.attention(x)
        #print(['the shape of x',tf.shape(x)])
        return x



def Contents(model,test_data):

    sub_model = Model(inputs=model.input,outputs=model.layers[-7].output)
    result = sub_model.predict(test_data,batch_size=100)

    scio.savemat('Urban_latent.mat',{'latent': result})

def Bilinear_Part(model,test_data):

    from numpy import savetxt  # Save as human readable
    sub_model = Model(inputs=model.input,outputs=model.layers[-3].output)
    result = sub_model.predict(test_data,batch_size=100)[1]
    savetxt('Bilinear.csv', result, delimiter=',')

def Abund_joint(model,test_data):

    from numpy import savetxt  # Save as human readable
    sub_model = Model(inputs=model.input,outputs=model.layers[-3].output)
    result = sub_model.predict(test_data,batch_size=100)[2]
    #result = result[0:10,:]
    savetxt('Abund_joint.csv', result, delimiter=',')

def Endmembers(trained_weights):

    import h5py
    keys = []
    with h5py.File(trained_weights, 'r') as f:
        f.visit(keys.append)
        # for key in keys:
        #     if ':' in key:
        #         print(f[key].name)
        #         print(f[key].value)
        print('gamma',f['/BMM/BMM/gamma:0'].value)
    key = '/BMM/BMM/EMs:0'
    f = h5py.File(trained_weights, 'r')
    group = f[key]
    #print(group)
    # b4 = group['/optimizer_weights/training/Adam/endmember/kernel/m:0'].value
    k4 = group.value
    k4 = np.transpose(np.array(k4))  # Process
    #print(k4)
    f.close()  # Close key
    plt.plot(k4)
    plt.show()
    from numpy import savetxt  # Save as human readable
    #open('w4ss.csv','wt').close()
    savetxt('w4ss.csv', k4, delimiter=',')
    print('w4ss.csv')

@tf.function
def SAD(y_pred, y_true):

    y_pred = tf.nn.l2_normalize(y_pred, axis=-1)
    y_true = tf.nn.l2_normalize(y_true, axis=-1)
    SAM = tf.reduce_sum(y_pred*y_true, axis=-1)
    SAM = tf.acos(1-SAM)
    return SAM

@tf.function
def cal_coeff(EMs):
        sum = 0
        for i in range(EMs.shape[0] - 1):
            for j in range(i + 1, EMs.shape[0] - 1):
                sum = sum + SAD(EMs[i, :], EMs[j, :])
        return sum

@tf.function
def diff_constraints(data):

    index = []
    for i in range(data.shape[1] - 1):
        index.append(i)
    datas = tf.gather(data, index, axis=1)
    first_col = tf.expand_dims(data[:, 0], axis=1)
    datas = tf.concat([first_col, datas], axis=1)
    diff = tf.reduce_sum(tf.abs(data - datas))

    return diff

class em_amplitude_regularization(regularizers.Regularizer):

    def __init__(self):
        super(em_amplitude_regularization, self).__init__()

    def __call__(self, x):

        neg = tf.cast(x<0, x.dtype) * x
        great_one = tf.cast(x>=1.0, x.dtype) * x

        return -10 * tf.reduce_sum(neg) + 10 * tf.reduce_sum(great_one)

class BMM(layers.Layer):
    def __init__(self,input_dim,out_dim,init,mode,**kwargs):
        super(BMM, self).__init__(**kwargs)
        #self.kernel = self.add_variable(name='w',shape=[input_dim,out_dim])
        self.input_dim = input_dim
        self.joint_num = input_dim*(input_dim-1) // 2
        self.out_dim = out_dim
        self.init = init
        self.mode = mode

    def coeff_regularization(self, x):

        return 0.001*cal_coeff(x)

    ##constraint the ems between zero and one, minimize the em<0 and em>0
    def em_regulization(self,x):

        neg_x = tf.cast(x<0,x.dtype) * x
        great_one = tf.cast(x>=1.0,x.dtype) *x
        return -10 * tf.reduce_sum(neg_x) + 10 * tf.reduce_sum(great_one)

    def smoothness_constraint(self, data):
        return 0.00001*diff_constraints(data)

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def build(self, input_shape):

        self.EMs = self.add_weight(name='EMs',shape=[input_shape[1],self.out_dim],initializer=self.init,regularizer=em_amplitude_regularization(),
                                  trainable=self.mode)
        self.gamma = self.add_weight(name='gamma', shape=[1,self.joint_num],initializer=initializers.random_uniform(0.0,1.0),
                                      constraint=BMM_gamma_constraint(min_value=0.0,max_value=1.0), trainable=self.mode)
        self.built = True

    def call(self, inputs, **kwargs):

        out1 = inputs @ self.EMs
        self.add_loss(self.smoothness_constraint(self.EMs))
        index1 = []
        index2 = []
        for i in range(self.input_dim-1):
            for j in range(i+1,self.input_dim):
                index1.append(i)
                index2.append(j)
        EMs_joint = tf.gather(self.EMs,index1,axis=0) * tf.gather(self.EMs,index2,axis=0)
        #print(EMs_joint.shape)
        Abund_joint = tf.gather(inputs, index1,axis=-1) * tf.gather(inputs, index2,axis=-1)
        # print(Abund_joint.shape)
        # print(self.gamma.shape)
        #EMs_joint = self.gamma * EMs_joint
        Abund_joint = Abund_joint * self.gamma
        out2 = Abund_joint @ EMs_joint
        out = out1 + out2

        return out,out2



