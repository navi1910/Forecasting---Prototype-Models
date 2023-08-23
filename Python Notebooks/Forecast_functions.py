def imports(proceed = True):
    if proceed == True:
        import pandas as pd
        import numpy as np

        from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, Dropout
        from tensorflow.keras.optimizers import Adam
        from sklearn.preprocessing import MinMaxScaler

def select_df(File_path, ProductCode, PlantCode, Sales_Channel, freq, interpolate = True, Dates_col = True):
    
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv(File_path, parse_dates=[0])
    
    df.index = df['Date']
    df = df.sort_index()
    
    df = df[(df['ProductCode'] == ProductCode) & (df['PlantCode'] == PlantCode)& (df['Sales_Channel']== Sales_Channel)]
    df = df.resample(freq).mean()
    
    if interpolate == True:
        df = df.interpolate(method = 'spline', order = 2)
    
    if Dates_col == True:
        df['Dates'] = df.index

    return df

def orders(data, lags, duration):
   
    from statsmodels.tsa.seasonal import seasonal_decompose
    from pandas.plotting import autocorrelation_plot
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    seasonal_decompose(data.iloc[:duration], model = 'additive').plot()
    
    plot_acf(data, lags = lags)
    
    plot_pacf(data, lags = lags)
    
    print('ACF is for MA - q, PACF is for AR - p and Trend is for d - p, d, q')

def train_test_split(data, train_size):
    
    train_size = int(data.shape[0]*train_size)
    
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    
    return train, test


def differencing(data, lag):
    data = data - data.shift(lag)
    
    return data.iloc[lag:]

def arima(data, p, d, q, summary = False):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(data, order = (p, d, q)).fit()
    
    if summary == True:
        print(model.summary())
    
    return model

def model_pred(model, train, test):
    
    pred = model.predict(start = len(train), end = len(train)+len(test)-1)
    
    return pred

def metrics(actual, pred, mae = True, mape = True, mse = True, plot = True):
    import numpy as np
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
    if mae == True:
        print('Mean Absolute Error: ', mean_absolute_error(actual, pred))
    if mape == True:
        print('Mean Absolute Percentage Error: ', mean_absolute_percentage_error(actual, pred))
    if mse == True:
        print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(actual, pred)))
        
    if plot == True:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 7))
        plt.plot(actual, label = 'Actual', c = 'g', alpha = 0.5, ls = '--')
        plt.plot(pred, label = 'Predictions', c = 'r', alpha = 0.5)
        plt.legend()


def sarimax(data, p, d, q, m, summary = False, ):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(data, order = (p, d, q), seasonal_order=(1,1,1,m)).fit()
    
    if summary == True:
        print(model.summary())
    
    return model

def lstm_rnn(df, col_name, date_col, train_size, plot = True, metrics = True):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    
    dataSize = len(df)
    df.head()
    
    dataMat = df[col_name].values.reshape(dataSize,1)
    dataMat.shape
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaledDataMat = scaler.fit_transform(dataMat)
    
    train_split = train_size

    train_size = int(train_split * dataSize)

    trainData = scaledDataMat[:train_size]
    valData = scaledDataMat[train_size:]

    print('Train data size: ', trainData.shape)
    print('Val data size: ', valData.shape)
    
    def createDataset(dataset, time_step=1):
        dataX, dataY = [], []
    
        for i in range(len(dataset)-time_step):
            dataX.append(dataset[i:(i+time_step)])
            dataY.append(dataset[i + time_step])
        return np.array(dataX), np.array(dataY)

    n_past = 4

    X_train,y_train = createDataset(trainData, time_step=n_past)
    X_val,y_val = createDataset(valData, time_step=n_past)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(n_past,1)))
    model.add(LSTM(64, return_sequences=False, input_shape=(n_past,1)))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error',optimizer='adam')
    history = model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=50,batch_size=1,verbose=1, callbacks=[callback])
    
    y_train_hat = scaler.inverse_transform(model.predict(X_train))

    y_val_hat = scaler.inverse_transform(model.predict(X_val))
    
    train_shift = n_past

    test_shift = train_shift + len(y_train_hat) + n_past
    
    if plot == True:
        plt.figure(figsize = (10,6))

        plt.plot(df[date_col], df[col_name], color='g', label="Original Data", alpha = 0.5, ls = '--')
        plt.plot(df[date_col][train_shift: test_shift-n_past], y_train_hat, color='r', label="Train data Pred", alpha = 0.8)
        plt.plot(df[date_col][test_shift:], y_val_hat, color='y', label="Val Data Pred", alpha = 0.8)

        plt.legend()
    
    if metrics == True:
        from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
        
        def metrics(actual, pred, mae = True, mape = True, mse = True, plot = True):
            if mae == True:
                print('Mean Absolute Error: ', mean_absolute_error(actual, pred))
            if mape == True:
                print('Mean Absolute Percentage Error: ', mean_absolute_percentage_error(actual, pred))
            if mse == True:
                print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(actual, pred)))
        
            if plot == True:
                import matplotlib.pyplot as plt
        
                plt.figure(figsize=(10, 7))
                plt.plot(actual, label = 'Actual', c = 'g', alpha = 0.5, ls = '--')
                plt.plot(pred, label = 'Predictions', c = 'r', alpha = 0.5)
                plt.legend()
        
        metrics(scaler.inverse_transform(y_val), y_val_hat)
        
    print()
    
    return model,scaler, y_val, y_val_hat




