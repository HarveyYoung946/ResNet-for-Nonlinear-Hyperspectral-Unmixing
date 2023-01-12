# coding = utf-8
from tensorflow.keras import layers, models, optimizers, callbacks, initializers, constraints, regularizers
import tensorflow as tf
import time
from Res_Netxt_DenseNet import Normalize, VCA_Initializer, \
    Extract_Features_Refs,  My_Initializer, Endmembers, \
    Contents, bias_contraints_MinMax,BMM,filter_data,Bilinear_Part,Abund_joint
from tensorflow.python.keras import backend as K
from libtiff import TIFF
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# def Ref_std(y_true,y_pred):
#
#     y_pred = tf.squeeze(y_pred, -1)
#     y_true = tf.squeeze(y_true, -1)
#     y_pred = tf.transpose(y_pred) / tf.reduce_max(tf.transpose(y_pred),axis=0)
#     y_pred = tf.transpose(y_pred)
#     y_pred = tf.expand_dims(y_pred,axis=-1)
#     y_true = tf.transpose(y_true) / tf.reduce_max(tf.transpose(y_true), axis=0)
#     y_true = tf.transpose(y_true)
#     y_true = tf.expand_dims(y_true,axis=-1)
#     return y_pred,y_true
#
# def JS_div(y_true,y_pred):
#     y_pred = tf.squeeze(y_pred, -1)
#     y_true = tf.squeeze(y_true, -1)
#     vec1 = y_true / tf.reduce_sum(y_true,axis=-1,keepdims=True)
#     vec2 = y_pred / tf.reduce_sum(y_pred,axis=-1,keepdims=True)
#     # KL = tf.reduce_sum(vec2 * tf.math.log(K.epsilon() + vec2/(vec1+K.epsilon())),axis=-1,keepdims=True)
#     # KL = tf.reduce_mean(KL)
#     JS1 = tf.reduce_sum(vec1*tf.math.log(K.epsilon()+vec1/((vec1+vec2)/2+K.epsilon())),axis=-1,keepdims=True)
#     JS2 = tf.reduce_sum(vec2*tf.math.log(K.epsilon()+vec2/((vec1+vec2)/2+K.epsilon())),axis=-1,keepdims=True)
#     JS = (JS1 + JS2)/2
#     return tf.reduce_mean(JS)
# # alpha = 0.15
# def Kl_for_log_probs(p, q):
#     # print(['the shape of the p:', p.shape])
#     log_p = tf.nn.log_softmax(p)
#     log_q = tf.nn.log_softmax(q)
#     p = tf.exp(log_p)
#     q = tf.exp(log_q)
#     p_logp = tf.reduce_sum(p * log_p, axis=-1)
#     # print(['the shape of the p:', p_logp.shape])
#     p_logq = tf.reduce_sum(p * log_q, axis=-1)
#     kl_p = p_logp - p_logq
#     q_logq = tf.reduce_sum(q * log_q, axis=-1)
#     q_logp = tf.reduce_sum(q * log_p, axis=-1)
#     kl_q = q_logq - q_logp
#     sum_kl = tf.reduce_sum(kl_p + kl_q)
#     # print(['the shape of the kl:',sum_kl.shape])
#     return sum_kl


alpha = 0.01


def LoadData(path_data):
    tif = TIFF.open(path_data, mode='r')
    img = tif.read_image()
    scale = np.max(img)
    img = np.array(img) / scale
    # #img = np.clip(img, 0.00001, 0.999999)
    img = np.transpose(img)
    img = np.abs(img)
    img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    # 过滤数据
    #img = filter_data(img, kernel_size=5, strides=5)
    # original ref train_data1
    data_stack = img.reshape(img.shape[0], img.shape[1], 1)
    label = img.reshape(img.shape[0], img.shape[1], 1)

    return data_stack, label,scale

@tf.function
def cosine_pro(y_pred, y_true):
    eps = 1e-6
    y_pred = tf.squeeze(y_pred, -1)
    y_true = tf.squeeze(y_true, -1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    y_true = K.l2_normalize(y_true, axis=-1)
    SAM = tf.reduce_sum(y_pred*y_true, axis=-1)
    SAM = tf.reduce_mean(tf.acos(SAM-eps))

    return SAM

def WbceLoss(y_true, y_pred):

        #复杂损失函数的调用
        cos_loss = cosine_pro(y_pred, y_true)
        mse_loss = tf.reduce_mean(tf.square(y_true-y_pred))
        cos_wmse_loss = cos_loss + mse_loss

        return  cos_wmse_loss

def unmixing_wbce(inputs_shape, n_endmembers, train_mode):

    x_orig_1 = layers.Input(shape=inputs_shape)
    x_weights = layers.Input(shape=inputs_shape)
    x_true = layers.Input(shape=inputs_shape)

    x_1 = Extract_Features_Refs(filter_num=32, kernel_init=initializers.he_uniform())(x_orig_1)
    x_1 = layers.Flatten()(x_1)
    x_1 = layers.Dense(n_endmembers, name='Dense_1', kernel_initializer=initializers.glorot_uniform())(x_1)
    x_1 = layers.BatchNormalization()(x_1)
    Abund = Normalize(min_value=0.05)(x_1)
    y_decoder,Bi_part = BMM(input_dim=n_endmembers, out_dim=x_orig_1.shape[1],name='BMM',
                    init=VCA_Initializer(data=train_data1,num=n_class),
                    mode=train_mode)(Abund)
    y_decoder = layers.Reshape(target_shape=[x_orig_1.shape[1], 1])(y_decoder)


    model = models.Model(x_orig_1, y_decoder)

    return model



# def train(model,train_data,args):
def train(model, train_data, label, args):

    # callbacks
    log = callbacks.CSVLogger('./log.csv')
    tb = callbacks.TensorBoard(log_dir='./logs', update_freq=batchsz, histogram_freq=1)
    checkpoint = callbacks.ModelCheckpoint(save_path+'-{epoch:02d}.h5', monitor='val_loss', save_best_only=False, save_weights_only=True, verbose=1, period=1)
    #checkpoint = callbacks.ModelCheckpoint('./Urban_test_weights-{epoch:02d}.h5', monitor='val_loss', save_best_only=False,save_weights_only=True, verbose=1, period=2)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch), verbose=1)

    #optimizer = optimizers.Adam(lr=args.lr)
    optimizer = optimizers.Nadam(lr=args.lr)

    model.compile(optimizer=optimizer,
                  loss= WbceLoss)

    model.fit(train_data, label, batch_size=batchsz, epochs=epochs, verbose=2, callbacks=[log, tb, checkpoint, lr_decay],validation_split=0.1,shuffle=True)


def test(model, test_data):

    Contents(model=model, test_data=test_data)
    Bilinear_Part(model=model,test_data=test_data)
    #Abund_joint(model=model,test_data=test_data)
    label = np.squeeze(test_data,-1)
    # 波长索引，用于画图
    wavelength = np.linspace(1, test_data.shape[1], test_data.shape[1])
    td = model.predict(test_data, batch_size=100)

    fig = plt.figure()
    for i in range(0, 8):
        plt.subplot(421 + i)
        plt.plot(wavelength, td[i, :])
        plt.plot(wavelength, label[i, :])
    plt.legend(['td', 'label'])
    plt.show()

batchsz = 512
epochs = 500

if __name__ == '__main__':

    import argparse

    # set the hyper parameters
    parser = argparse.ArgumentParser(description='RDAE Network on HSI')
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--lr_decay', default=1, type=float)
    parser.add_argument('--debug', action='store_true', help='Save  weights by TensorBoard')
    args = parser.parse_args()

    ##Samson
    n_class = 3
    path_data = 'samson_ori.tif'
    save_path = 'samson_weights'

    #Jasper
    # n_class = 4
    # path_data = 'Jasper_Ridge_ori.tif'
    # save_path = 'Jasper_Ridge_weights'

    ##Urban
    # n_class = 4
    # path_data = 'Urban_R162.tif'
    # save_path = 'urban_weights'

    train_data1, labels, scale = LoadData(path_data)
    #img_max = np.max(train_data 1)
    print(['the shape of the train_data',train_data1.shape,'the max of the train_data',scale])


   # test=1 represents training and test=0 represents testing
    train_or_test = 1

    start_time = time.time()
    if train_or_test == 1:
        model = unmixing_wbce(inputs_shape=[train_data1.shape[1], 1], n_endmembers=n_class, train_mode=True)
        model.summary()

        train(model=model, train_data=train_data1, label = labels, args=args)

    else:
        model = unmixing_wbce(inputs_shape=[train_data1.shape[1], 1], n_endmembers=n_class, train_mode =False)

        trained_weights = save_path+'-500.h5'
        #model.load_weights(trained_weights,by_name=True)
        model.load_weights(trained_weights)

        Endmembers(trained_weights=trained_weights)
        model.summary()
        test(model=model, test_data=train_data1)

        print('done')
    end_time = time.time()
    print('running time is: ',end_time-start_time)
