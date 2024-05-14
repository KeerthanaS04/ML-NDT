w=256
h=256

input_tensor = Input(shape=(w,h,1))

# start with max-pool to envelop the UT-data
h = layers.MaxPooling2D(pool_size=(window,1),  padding='valid' )(input_tensor) # MaxPooling1D would work, but we may want to pool adjacent A-scans in the future

#build the network
h=Conv2D(128,(1,1),padding='same',activation='relu')(h)
h=BatchNormalization()(h)
h=Activation('relu')(h)
h=Conv2D(128,(3,3))(h)
h=BatchNormalization()(h)
#     h=MaxPooling2D((2,2),strides=(2,2))(h)
h=Activation('relu')(h)

h=SeparableConv2D(94,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h=Activation('relu')(h)
# h=SeparableConv2D(64,(3,3),padding='same')(h)
# h=BatchNormalization()(h)
# h=Activation('relu')(h)
h=MaxPooling2D((2,2),strides=(2,2))(h)

h=SeparableConv2D(94,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h=Activation('relu')(h)
#     h=SeparableConv2D(128,(3,3),padding='same')(h)
#     h=BatchNormalization()(h)
#h=MaxPooling2D((2,2),strides=(2,2))(h)

# h=SeparableConv2D(128,(3,3),padding='same')(h)
# h=BatchNormalization()(h)
# h=Activation('relu')(h)
h=SeparableConv2D(64,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h=Activation('relu')(h)
h=MaxPooling2D((2,2),strides=(2,2))(h)

# h=SeparableConv2D(256,(3,3),padding='same')(h)
# h=BatchNormalization()(h)
# h=Activation('relu')(h)
# h=SeparableConv2D(512,(3,3),padding='same')(h)
# h=BatchNormalization()(h)
# h=Activation('relu')(h)
# h=SeparableConv2D(512,(3,3),padding='same')(h)
# h=BatchNormalization()(h)
# h=Activation('relu')(h)

# h=SeparableConv2D(512,(3,3),padding='same')(h)
# h=BatchNormalization()(h)
# h=Activation('relu')(h)
# h=SeparableConv2D(512,(3,3),padding='same')(h)
# h=BatchNormalization()(h)
# h=Activation('relu')(h)
# h=MaxPooling2D((2,2),strides=(2,2))(h)

# h=SeparableConv2D(512,(3,3),padding='same')(h)
# h=BatchNormalization()(h)
# h=Activation('relu')(h)
# h=SeparableConv2D(512,(3,3),padding='same')(h)
# h=BatchNormalization()(h)
# h=Activation('relu')(h)
# h=MaxPooling2D((2,2),strides=(2,2))(h)

# h=SeparableConv2D(512,(3,3),padding='same')(h)
# h=BatchNormalization()(h)
# h=Activation('relu')(h)
# h=SeparableConv2D(256,(3,3),padding='same')(h)
# h=BatchNormalization()(h)
# h=Activation('relu')(h)
# h=SeparableConv2D(256,(3,3),padding='same')(h)
# h=BatchNormalization()(h)

# h=Activation('relu')(h)
h=SeparableConv2D(64,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h=Activation('relu')(h)
h=SeparableConv2D(32,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h=Activation('relu')(h)

h=SeparableConv2D(32,(3,3),padding='same')(h)
h=BatchNormalization()(h)
h=Activation('relu')(h)

h=GlobalAveragePooling2D()(h)

h=Dense(16,activation='relu')(h)
h=Dropout(0.4)(h)
h=Dense(16,activation='relu',name='RNN')(h)
h=Dropout(0.3)(h)
h=Dense(1,activation='sigmoid')(h)

outputs=h
    
# model=Model(inputs=inputs,outputs=outputs)


model = Model(input_tensor, outputs)
opt = keras.optimizers.Adam(lr=0.0001, clipnorm=1.)
model.compile(optimizer=opt, loss='binary_crossentropy' , metrics=['acc'])
model.summary()
