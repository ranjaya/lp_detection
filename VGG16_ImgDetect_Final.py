from keras.utils.io_utils import HDF5Matrix

from keras.applications import VGG16
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Flatten

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras import backend as K

#jumlah image best sejauh ini = 2260
#   training = 70%  -> 1582
#   testing = 15%   -> 339
#   validation = 15%-> 339


def lp_train_generator():
    i=0
    img_data = HDF5Matrix('C:/Users/Ricky/PycharmProjects/untitled/lp_train.h5','images')
    lbl_data = HDF5Matrix('C:/Users/Ricky/PycharmProjects/untitled/lp_train.h5','labels')
    size = 1582

    while 1:
        img_single = img_data[i % size].reshape((1,320,240,3))
        lbl_single = lbl_data[i % size].reshape((1,4))
        # img_single = img_data[i % size].reshape((320, 240, 3))
        # lbl_single = lbl_data[i % size].reshape(4)
        yield(img_single, lbl_single)
        i+=1


def lp_valid_generator():
    i=0
    img_data = HDF5Matrix('C:/Users/Ricky/PycharmProjects/untitled/lp_valid.h5','images')
    lbl_data = HDF5Matrix('C:/Users/Ricky/PycharmProjects/untitled/lp_valid.h5','labels')
    size = 339

    while 1:
        img_single = img_data[i % size].reshape((1,320,240,3))
        lbl_single = lbl_data[i % size].reshape((1,4))
        # img_single = img_data[i % size].reshape((320, 240, 3))
        # lbl_single = lbl_data[i % size].reshape(4)
        yield(img_single, lbl_single)
        i+=1


def lp_test_generator():
    i=0
    img_data = HDF5Matrix('C:/Users/Ricky/PycharmProjects/untitled/lp_test.h5','images')
    lbl_data = HDF5Matrix('C:/Users/Ricky/PycharmProjects/untitled/lp_test.h5','labels')
    size = 339

    while 1:
        img_single = img_data[i % size].reshape((1,320,240,3))
        lbl_single = lbl_data[i % size].reshape((1,4))
        # img_single = img_data[i % size].reshape((320, 240, 3))
        # lbl_single = lbl_data[i % size].reshape(4)
        yield(img_single, lbl_single)
        i+=1


def iou(y_true,y_pred):

    print("IoU function . . .")

    # set value for x,y,w,h of box 1 and box 2
    x1 = y_pred[0][0]
    y1 = y_pred[0][1]
    w1 = y_pred[0][2]
    h1 = y_pred[0][3]

    x2 = y_true[0][0]
    y2 = y_true[0][1]
    w2 = y_true[0][2]
    h2 = y_true[0][3]

    print('x1 = ', x1)

    #convert the value of each variable from normalize form to integer
    min = K.cast(0, dtype='float32')
    max_xw = K.cast(320, dtype='float32')
    max_yh = K.cast(240, dtype='float32')

    x1, x2 = x1 * max_xw, x2 * max_xw
    y1, y2 = y1 * max_yh, y2 * max_yh
    w1, w2 = w1 * max_xw, w2 * max_xw
    h1, h2 = h1 * max_yh, h2 * max_yh

    #calculation for IoU
    and_x1, and_y1 = K.maximum(x1,x2), K.maximum(y1,y2)
    and_x2, and_y2 = K.minimum(x1,x2), K.minimum(y1,y2)

    and_w = and_x2 - and_x1
    and_h = and_y2 - and_y1

    # if K.less_equal(and_w, min) and K.less_equal(and_h, min):
    #     return K.cast(0, dtype='float32')
    and_area = and_w * and_h
    area1 = w1*h1
    area2 = w2*h2
    or_area = area1 + area2 - and_area

    return and_area / or_area


if __name__ == '__main__':

    w, h = 320, 240
    num_classes = 4

    nb_epoch = 100
    init_epoch = 0

    batch_size = 16

    # lr = 1e-03
    lr = 1e-04

    train_size = 1582
    valid_size = 339
    test_size = 339


    ###model vgg
    base_model = VGG16(input_shape=(320,240,3),include_top=False,classes=num_classes)
    x = base_model.get_layer('block5_pool').output
    flatten = Flatten(name='flatten')(x)
    # detection = Dense(4, activation='sigmoid', name='fcnew1')(flatten)
    #detection = Dense(2048, activation='relu', name='fcnew1')(flatten)
    #new - add 1 more dense
    # detection = Dense(4, activation='sigmoid', name='fcnew2')(detection)
    #detection = Dense(2048, activation='relu', name='fcnew2')(detection)
    #new - add 1 more dense
    detection = Dense(4, activation='sigmoid', name='fcnew')(detection)

    #create custom vgg
    custom_model = Model(inputs = base_model.input, outputs = detection)
    print(custom_model.summary())

    #freeze model
    for layer in custom_model.layers[:-1]:
        layer.trainable = False

    for n in custom_model.layers:
        print(n, n.trainable)

    print(custom_model.summary())


    ###optimizer
    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    custom_model.compile(optimizer=opt, loss='mse', metrics=[iou])

    checkpoint = ModelCheckpoint("license_plate_best_weight.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger('license_plate.csv', append=False)
    tensorboard = TensorBoard(log_dir='./tf-logs-licenseplate')
    callback_list = [checkpoint, csv_logger, tensorboard]

    result = custom_model.fit_generator(lp_train_generator(),
            train_size // batch_size,
            1000,
            # init_epoch = 0,
            validation_data=lp_valid_generator(),
            validation_steps=valid_size//batch_size,
            shuffle=True,
            callbacks=callback_list
            )

    accuracy = custom_model.evaluate_generator(lp_test_generator(),steps=test_size//batch_size)
