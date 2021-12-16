import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input,LSTM
from tensorflow.keras.optimizers import Adam

#NMAE function
def nmae_get(y, y_hat):
    y_av = np.mean(y)
    y_sum = np.sum(np.abs(y - y_hat))
    return y_sum/(len(y)*y_av)
    
#NMAE for h=0:::10
def nmaes_array(df_test, df_pre, h):
    nmaes = []
    for i in range(0, h+1):
        y_predict_i = df_pre[:, i]
        y_test_o = df_test.iloc[:, i].to_numpy()
        nmaes.append(nmae_get(y_test_o, y_predict_i))
        
    return nmaes
    
def Rnn_reshape(samples, time_step, features):
    X = samples.to_numpy()
    X = X.reshape(X.shape[0], time_step, features)
    return X
    
def LSTM_model(input_X, input_Y, l, h, n_features):
    X = Rnn_reshape(input_X, l, n_features)
    
    model = Sequential()
    model.add(LSTM(50, activation='relu',dropout=0.3,return_sequences=True, input_shape=(l, n_features)))
    model.add(LSTM(50, activation='relu',dropout=0.2))
    model.add(Dense(h))
    opt = Adam(lr=0.01)
    model.compile(optimizer=opt, loss='mae',metrics=['accuracy'])
    model.fit(X, input_Y, epochs=50, verbose=1, batch_size=32,validation_split=0.2)
    return model
    
def fetch_data(data_X,data_Y,h,l):
    num_range = len(data_X)-1*l-1*h
    X = data_X[0:num_range]
    Y = data_Y[l*1:l*1+num_range]['ReadsAvg']
    target = ['ReadsAvg']
    index = X.index
    Y.index = index
    feature = [col for col in data_X] 
    
    for i in range(l):
        if(i==0):
            X = X
        if(i>0):
            X_add = data_X[(i+1)*1:(i+1)*1+num_range]
            X_add.columns = [j+str(i+2) for j in feature] 
            X_add.index = index
            X = pd.concat([X,X_add],axis=1)
    
    for i in range(h):
        if(i==0):
            Y = Y
        if(i>0):
            Y_add = data_Y[(l+i+1)*1:(l+i+1)*1+num_range]['ReadsAvg']
            Y_add.columns = [j+str(i+2) for j in target] 
            Y_add.index = index
            Y = pd.concat([Y,Y_add],axis=1)
    return X,Y

X_latest = pd.read_excel('Xlatest.xlsx', index_col=0) 
Y_latest = pd.read_excel('Ylatest.xlsx', index_col=0) 

X_train, X_test, Y_train, Y_test = train_test_split(X_latest, Y_latest, test_size=0.3, shuffle = False)
print(X_train.shape,"(70% of the samples in training set and 16 features)")
X_train = X_train.sort_index(axis = 0)
X_test = X_test.sort_index(axis = 0)
Y_train = Y_train.sort_index(axis = 0)
Y_test = Y_test.sort_index(axis = 0)

#Train 64 values

nmaesdf = pd.DataFrame()
for l in range(0, 65):
    Rnn_xtrain, Rnn_ytrain = fetch_data(X_train, Y_train, 65, l+1)
    Rnn_xtest, Rnn_ytest = fetch_data(X_test, Y_test, 65, l+1)
    Rnn_model = LSTM_model(Rnn_xtrain, Rnn_ytrain, l+1, 65, 16)
    Rnn_test = Rnn_reshape(Rnn_xtest, l+1, 16)
    Rnn_pred = Rnn_model.predict(Rnn_test,verbose=0)
    nmaes_l = nmaes_array(Rnn_ytest, Rnn_pred, 64)
    nmaesdf['nmaes_l'+str(l)] = nmaes_l
    print(l)
    
nmaesdf.to_excel("nmaesdf64.xlsx")

