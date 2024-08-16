# Product Sales Data Analiz
 Bu veri seti, bir işletmenin 2019'dan 2023'e kadar satış işlemlerine ilişkin bilgileri içermektedir. Her bir satırda, belirli bir ürüne ait satışın temel detayları yer almaktadır. Bu veri seti, ürün kimliklerinden satış tarihine kadar çeşitli bilgiler içermektedir.(Bu veri Chatgpt tarafından yapılmıştır.)                                      
Sütunlar:                                                          
1) Urun kimligi:                                                       
Satışı yapılan ürünün kimlik numarası. Bu sütun, her bir ürünü benzersiz bir şekilde tanımlar.
2) Satilan urunun adi:                                            
Satılan ürünün adı. Bu sütun, hangi ürünün satıldığını belirtir.
3) Urunun satildigi tarih:                                  
Ürünün satıldığı tarih. Bu sütun, satış işleminin ne zaman gerçekleştiğini gösterir.
Satilan urun miktari:                                         
4) Satılan ürünün miktarı. Bu sütun, satış işlemi sırasında kaç adet ürün satıldığını belirtir.

## Kütüphaneler
İhtiyacımız olan kütüphaneler

```
import numpy as np
import pandas as pd, datetime
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import calendar
import joblib
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
from time import time
import os
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import  ARIMA
from pmdarima import auto_arima
import lightgbm as lgb
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from pandas import DataFrame
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from prophet import Prophet
import warnings
import pmdarima as pm
from collections import Counter
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
from flask import Flask, request, jsonify
```
## Veriyi Yükleme
```
data = pd.read_csv('C:/.../Product_Sales_Data.csv')
data['Satış Tarihi'] = pd.to_datetime(data['Satış Tarihi'], errors='coerce')
```
İlk 5 veriye baktık.
```
data.head()
```
![image](https://github.com/user-attachments/assets/b1394dda-2d03-46cd-a1b7-dfd32f54f90b)

Veri setimiz hakkında daha çok bilgi edindik.
```
data.info()
```
![image](https://github.com/user-attachments/assets/a5212c4d-533c-4581-ba6b-f504f624fb13)
Burda 2000 gözlemimizin olduğunu gördük yani eksik değer olmadığını gördük.

# Burda sütun adını değiştirmek istiyorum çünkü türkçe karekterler var ayrıca sütun başlıklarımda boşluklar vardı
```
data.rename(columns={'Ürün Adı': 'Urun_adı'}, inplace=True)
data.rename(columns={'Satış Tarihi': 'Satıs_Tarihi'}, inplace=True)
```

Kolonlara baktık.
```
data.columns
```
![image](https://github.com/user-attachments/assets/d638df3c-de42-4e7b-9a7f-9e8c23044c91)

## Veriyi analiz için uygun hale getirmek
```
data['Satıs_Tarihi'] = pd.to_datetime(data['Satıs_Tarihi'])
data['Yıl'] = data['Satıs_Tarihi'].dt.year
data['Ay'] = data['Satıs_Tarihi'].dt.month
data['Gun'] = data['Satıs_Tarihi'].dt.day
data['Hafta'] = data['Satıs_Tarihi'].dt.isocalendar().week
```
![image](https://github.com/user-attachments/assets/8591d59c-7749-41bc-b754-f5090f0f1a9c)

## Geçersiz Değerleri Bulma
Geçersiz değerlerimiz var mı kontrol ettik ve onları çıkardık.
```
print(data['Satıs_Tarihi'].head())
print(data['Satıs_Tarihi'].dtype)
print(data['Satıs_Tarihi'].unique())
data = data.sort_values(by='Satıs_Tarihi')
data['Satıs_Tarihi'] = pd.to_datetime(data['Satıs_Tarihi'], errors='coerce')

invalid_dates = data[data['Satıs_Tarihi'].isna()]
print(invalid_dates)

data = data.dropna(subset=['Satıs_Tarihi'])
```

## Görselleştirme
Burda aylar ve günlerimiz sayısal değerlere sahipti
```
sns.set(style="whitegrid")
month_map = {
    1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan',
    5: 'Mayıs', 6: 'Haziran', 7: 'Temmuz', 8: 'Ağustos',
    9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
}

day_map = {
    1: 'Pazartesi', 2: 'Salı', 3: 'Çarşamba', 4: 'Perşembe',
    5: 'Cuma', 6: 'Cumartesi', 7: 'Pazar'
}

def bar_plot(variable, use_names=False):
    var = data[variable]
    
    if use_names:
        if variable == 'Ay':
            var = var.map(month_map)  
        elif variable == 'Gun':
            var = var.map(day_map)  
    varValue = var.value_counts().sort_index()  
    
    plt.figure(figsize=(15,21))
    sns.barplot(x=varValue, y=varValue.index, orient='h')  # X eksenine frekanslar, Y eksenine ürün adları
    plt.title(f'{variable} dağılımı ({ "İsimler" if use_names else "Sayısal" })')
    plt.xlabel('Frekans')
    plt.ylabel(variable)
    plt.xticks(fontsize=9)  
    plt.yticks(fontsize=9)
    plt.show()


def visualize_all():
  bar_plot("Urun_adı")
  bar_plot('Yıl')
  bar_plot('Ay', use_names=True)
  bar_plot("Hafta")
  bar_plot('Gun', use_names=True)
```
```
def histogram_plot(variable):
    plt.figure(figsize=(10,6))
    sns.histplot(data[variable], bins=30, kde=True)
    plt.title(f'{variable} dağılımı')
    plt.xlabel(variable)
    plt.ylabel('Frekans')
    plt.show()
def box_plot(variable):
    plt.figure(figsize=(10,6))
    sns.boxplot(y=data[variable])
    plt.title(f'{variable} için Box Plot')
    plt.show()
histogram_plot('Satıs_Tarihi')
box_plot('Adet')
```
visualize_all()
![output_19_0](https://github.com/user-attachments/assets/c02ebc0d-d196-45f5-8a7a-ea0623bde567)
![output_19_1](https://github.com/user-attachments/assets/e16ee8ad-25d2-41a5-ad0f-7ae520269d66)![output_19_2](https://github.com/user-attachments/assets/04c7970c-16f4-4659-b117-516ab517046f)
![output_19_3](https://github.com/user-attachments/assets/c03be26b-4795-48d2-8acd-5dac0285c3a7)
![output_19_4](https://github.com/user-attachments/assets/22749691-9f7f-4214-85ef-c6b61debf173)
![output_19_5](https://github.com/user-attachments/assets/fa3a0fb1-5f10-49d9-b200-355334d8d35c)
![output_19_6](https://github.com/user-attachments/assets/a6cc77d0-7c88-496e-affe-b426b4c1e0e8)

## Eksik Değerler Analizi ve Doldurulması
Burda eksik değer kontrolü yaptık, yukarıdada onu görmüştük zaten fakat kontrol ettik.
```
data.isnull().sum()
```
Eksik değere rastlamadık.

## Aykırı Değer Analizi
```
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        Q1 = np.percentile(df[c],25)
        Q3 = np.percentile(df[c],75)
        IQR = Q3 - Q1
        outlier_step = IQR * 1.5
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
liste = ["Adet"]
data.loc[detect_outliers(data,liste)]
```
![image](https://github.com/user-attachments/assets/893297de-54c2-45ea-99ff-2d75b37469d2)
Göründüğü üzere herhangi bir aykırı değere de rastlamadık.

## Basit EDA
Burda hangi üründen kaç tane satıldığının ortalama satış adetine baktık.
```
data[["Adet","Urun_adı"]].groupby("Urun_adı", as_index=True).mean().sort_values(by="Adet",ascending = False).head(10)
```
![image](https://github.com/user-attachments/assets/23b5178a-a1a3-44d3-93bc-1be324ee61dd)

İlk 10 tanesine baktığımızda Ürün_73, Ürün_28, Ürün_108, Ürün_90, Ürün_61, Ürün_122, Ürün_94, Ürün_93, Ürün_13, Ürün_117. Olduğunu gördük.

her yıl için ürünlerin ortalama satış adetlerini hesaplar. Bu ortalama, yıllık bazda ürünlerin ne kadar satıldığını gösterir.
```
data[["Adet","Yıl"]].groupby("Yıl", as_index=True).mean().sort_values(by="Adet",ascending = False)
```
![image](https://github.com/user-attachments/assets/4500a96f-8b58-4be6-9963-b02c13b8ddda)

yıllık ortalama satış adetleri üzerinden. En çok ortalama 2020 yılında daha sonrasında 2023 yılında yapıldığını ve en az 2021 yılında yapıldığını söyleyebiliriz.

Her ay için ürünlerin ortalama satış adetlerini hesaplar. Bu, her ay boyunca ürünlerin ne kadar satıldığını gösterir.
```
data[["Adet","Ay"]].groupby("Ay", as_index=True).mean().sort_values(by="Adet",ascending = False)
```
En Yüksek Satış Adedi:
Aralık (Ay 12), ortalama satış adedi 72.76 ile en yüksek satış performansını göstermektedir. Bu, yılın son ayında ürünlerin en yüksek satış rakamına ulaştığını gösterir. 

Ortalama Satışlar:
Nisan (Ay 4), 57.26 ortalama satış adedi ile ikinci sırada yer alıyor, Temmuz (Ay 7) ve Haziran (Ay 6) ise sırasıyla 57.31 ve 55.38 ile yüksek ortalamalara sahip. 

Düşük Satış Adedi:
Şubat (Ay 2), ortalama satış adedi 43.77 ile en düşük satış performansını gösteriyor. 

## Zaman Serisi Analizi Yapma
Burada otokorelasyon var mı verilerim acaba önceki verilerimden etkileniyor mu onu kontrol ediyoruz.
```
plot_acf(data['Adet'])
plt.show()

plot_pacf(data['Adet'])
plt.show()
```
![output_34_0](https://github.com/user-attachments/assets/bf2bb594-1447-48d3-bd6a-474025a63d3b)
![output_34_1](https://github.com/user-attachments/assets/fba727f7-5a21-4c4c-8586-b0bc7b07aeb5)

ACF ve PACF Grafiklerinde gördüğümüz üzere otokorelasyonumuz yok, bu yüzden ekstra bir düzenleme yapmamıza gerek yok.

## Haftalık Satışları Görselleştirme
```
hafta = data[['Satıs_Tarihi', 'Adet']].set_index('Satıs_Tarihi')
hafta = hafta.resample('W').mean()
hafta = hafta.fillna(hafta.mean())

plt.figure(figsize=(10, 5))
hafta.plot(ax=plt.gca())
plt.title('Haftalık Satışlar')
plt.xlabel('Tarih')
plt.ylabel('Satışlar')
plt.show()
```
![output_36_0](https://github.com/user-attachments/assets/75fa1a1e-290e-4f40-9e67-c2f5cfc45778)

Grafiğe baktığımızda çok belirgin bir mevsimsellik gözükmüyor satışlar rastgele yükselip azalmış gibi duruyor. Ayrıca belirgin bir trendimizde gözükmüyor.

## Haftalık Tahminler İçin Zaman Serisi Analizi
### En İyi Modeli Seçme (RMSE)
Burda birden çok model eğitiyoruz ve hatalarını karşılaştıracağız.
```
def calculate_rmse(actual, predicted):
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    return np.sqrt(mean_squared_error(actual[mask], predicted[mask]))

def auto_arima_forecast(data, date_column, value_column):
    data[date_column] = pd.to_datetime(data[date_column])
    train = data[data[date_column].dt.year < 2023]
    train_arima = train[[date_column, value_column]].set_index(date_column).resample('W').mean()
    
    train_arima = train_arima.fillna(train_arima.mean())

    model = pm.auto_arima(train_arima,
                          start_p=0, start_q=0,
                          test='adf', max_p=3, max_q=3,
                          m=52, start_P=0, seasonal=True,
                          d=1, D=1, trace=False, 
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    
    forecast = model.predict(n_periods=52)
    forecast_index = pd.date_range(start='2023-01-01', periods=52, freq='W')
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    rmse_ar = calculate_rmse(train_arima[-52:].values.flatten(), forecast_series.values)
    return rmse_ar

def xgboost_forecast(data, date_column, value_column):
    data[date_column] = pd.to_datetime(data[date_column])
    train = data[data[date_column].dt.year < 2023]
    train_weekly = train[[date_column, value_column]].set_index(date_column).resample('W').mean()

    train_weekly = train_weekly.fillna(train_weekly.mean())

    train_weekly['Week'] = train_weekly.index.isocalendar().week
    train_weekly['Month'] = train_weekly.index.month
    train_weekly['Year'] = train_weekly.index.year
    
    X_train = train_weekly[['Week', 'Month', 'Year']]
    y_train = train_weekly[value_column]
    
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
    model.fit(X_train, y_train)
    
    future_dates = pd.date_range(start='2023-01-01', periods=52, freq='W')
    future = pd.DataFrame({
        'Week': future_dates.isocalendar().week,
        'Month': future_dates.month,
        'Year': future_dates.year
    })

    forecast = model.predict(future)
    forecast_series = pd.Series(forecast, index=future_dates)
    
    rmse_xg = calculate_rmse(y_train[-52:].values, forecast_series.values)
    return rmse_xg

def ets_forecast(data, date_column, value_column):
    data[date_column] = pd.to_datetime(data[date_column])
    train = data[data[date_column].dt.year < 2023]
    train_weekly = train[[date_column, value_column]].set_index(date_column).resample('W-SUN').mean()

    train_weekly = train_weekly.fillna(train_weekly.mean())

    model = ExponentialSmoothing(train_weekly[value_column], trend='add', seasonal='add', seasonal_periods=52).fit()
    
    forecast = model.forecast(steps=52)
    forecast.index = pd.date_range(start='2023-01-01', periods=52, freq='W-SUN')
    
    rmse_ets = calculate_rmse(train_weekly[-52:].values.flatten(), forecast.values)
    return rmse_ets

def prophet_forecast(data, date_column, value_column, periods=52):
    data = data[[date_column, value_column]].rename(columns={date_column: 'ds', value_column: 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(data)
    
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    
    actual = data['y'].iloc[-periods:].values
    predicted = forecast['yhat'].iloc[-periods:].values
    
    rmse_pro = calculate_rmse(actual, predicted)
    return rmse_pro

def lightgbm_forecast(data, date_column, value_column):
    data[date_column] = pd.to_datetime(data[date_column])
    train = data[data[date_column].dt.year < 2023]
    train_weekly = train[[date_column, value_column]].set_index(date_column).resample('W').mean()

    train_weekly = train_weekly.fillna(train_weekly.mean())
    train_weekly['Week'] = train_weekly.index.isocalendar().week
    train_weekly['Month'] = train_weekly.index.month
    train_weekly['Year'] = train_weekly.index.year
    
    X_train = train_weekly[['Week', 'Month', 'Year']]
    y_train = train_weekly[value_column]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05, 'n_estimators': 1000}
    model = lgb.train(params, train_data)

    future_dates = pd.date_range(start='2023-01-01', periods=52, freq='W')
    future = pd.DataFrame({
        'Week': future_dates.isocalendar().week,
        'Month': future_dates.month,
        'Year': future_dates.year
    })

    forecast = model.predict(future)
    forecast_series = pd.Series(forecast, index=future_dates)
    
    rmse_lg = calculate_rmse(y_train[-52:].values, forecast_series.values)
    return rmse_lg

rmse_ar = auto_arima_forecast(data, 'Satıs_Tarihi', 'Adet')
rmse_xg = xgboost_forecast(data, 'Satıs_Tarihi', 'Adet')
rmse_ets = ets_forecast(data, 'Satıs_Tarihi', 'Adet')
rmse_pro = prophet_forecast(data, 'Satıs_Tarihi', 'Adet')
rmse_lg = lightgbm_forecast(data, 'Satıs_Tarihi', 'Adet')

print(f"Auto ARIMA RMSE: {rmse_ar}")
print(f"XGBoost RMSE: {rmse_xg}")
print(f"ETS RMSE: {rmse_ets}")
print(f"Prophet RMSE: {rmse_pro}")
print(f"LightGBM RMSE: {rmse_lg}")

models = ['Auto ARIMA', 'XGBoost', 'ETS', 'Prophet', 'LightGBM']
rmse_values = [rmse_ar, rmse_xg, rmse_ets, rmse_pro, rmse_lg]

plt.figure(figsize=(10, 6))
plt.bar(models, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Model Karşılaştırması: RMSE Değerleri')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```
Auto ARIMA RMSE: 23.023238985767765
XGBoost RMSE: 30.260661155892155
ETS RMSE: 19.043500044749916
Prophet RMSE: 39.64563233240264
LightGBM RMSE: 21.186319255100354

![output_43_2](https://github.com/user-attachments/assets/f6e44965-b910-46a8-90e8-1c4e5cf45cbb)

Görünüşe göre en iyi modelimiz ETS modelimiz geldi fakat verimizde belirgin bir trend veya mevsimsellik yok demiştik. Bu yüzden Auto Arima ve LGBM modelleri kurulabilir hataları birbirlerine yakın oldukları için ben auto arimadan gideceğim.

### Modeli Geliştirme Grafiklere Bakma
Veri setimde belirli bir mevsimsellik veya trend olmadığı için makine LGBM ve arima yapabiliriz, bunları yapmamız daha mantıkı olabilir, LGBM yaptığımızda modelimiz öğrenemiyor düzgün tahminler yapamıyor o yüzden Arima modelden devam ettik ve modelin performansı yükseltmeye çalıştık.

```
def auto_arima_forecast(data, date_column, value_column):
    data[date_column] = pd.to_datetime(data[date_column])
    train = data[data[date_column].dt.year < 2023]
    train_arima = train[[date_column, value_column]].set_index(date_column).resample('W').mean()
    
    train_arima = train_arima.fillna(train_arima.mean())

    model = pm.auto_arima(
        train_arima,
        start_p=0, start_q=0,
        max_p=5, max_q=5, 
        seasonal=True, m=52,
        start_P=0, start_Q=0,
        max_P=3, max_Q=3,  
        D=1,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,  
        n_jobs=1  
    )

    forecast = model.predict(n_periods=52)
    forecast_index = pd.date_range(start='2023-01-01', periods=52, freq='W')
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    actual_values = train_arima[-52:].values.flatten()
    forecast_values = forecast_series.values
    
    if len(actual_values) != len(forecast_values):
        min_len = min(len(actual_values), len(forecast_values))
        actual_values = actual_values[-min_len:]
        forecast_values = forecast_values[:min_len]
    
    rmse_ar = calculate_rmse(actual_values, forecast_values)
    
    aic = model.aic()
    bic = model.bic()
    
    print(f"En İyi Parametreler: {model.get_params()}")
    print(f"RMSE: {rmse_ar}")
    print(f"AIC: {aic}")
    print(f"BIC: {bic}")
    
    return forecast_series, forecast_index

forecast_series, forecast_index = auto_arima_forecast(data, 'Satıs_Tarihi', 'Adet')

plt.figure(figsize=(10, 7))
plt.plot(hafta.index, hafta['Adet'], label='Gerçek Veriler', color='royalblue')
plt.plot(forecast_index, forecast_series, label='Tahminler (ARIMA)', color='darkorange')
plt.fill_between(forecast_index, 
                 forecast_series - 1.96 * forecast_series.std(),
                 forecast_series + 1.96 * forecast_series.std(),
                 color='darkorange', alpha=0.2)

plt.title('Haftalık Satışlar ve Tahminler (ARIMA)')
plt.xlabel('Tarih')
plt.ylabel('Satışlar')
plt.legend()
plt.grid(True)
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2024-12-31'))
plt.show()
```
RMSE: 23.107664037987266
AIC: 1483.0383277364708
BIC: 1492.2070651525157
![image](https://github.com/user-attachments/assets/9cc85fb0-4315-47d8-866a-d1259eefe4a1)

Hiperparametre ayarlaması yaptığımızda maksimum performansımıza ulaşıyoruz. Bu yeterli diyebiliriz, grafiğe baktığımızda tahminlerim gerçek değelerlerle çok yüksek oranda eş gidiyor.

```
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def ets_forecast(data, date_column, value_column):
    data[date_column] = pd.to_datetime(data[date_column])
    train = data[data[date_column].dt.year < 2023]
    train_ets = train[[date_column, value_column]].set_index(date_column).resample('W').mean()
    
    train_ets = train_ets.fillna(train_ets.mean())

    model = ExponentialSmoothing(
        train_ets,
        trend='add',
        seasonal='mul',
        seasonal_periods=52,
        damped_trend=True
    ).fit()

    forecast = model.forecast(52)
    forecast_index = pd.date_range(start='2023-01-01', periods=52, freq='W')
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    actual_values = train_ets[-52:].values.flatten()
    forecast_values = forecast_series.values
    
    if len(actual_values) != len(forecast_values):
        min_len = min(len(actual_values), len(forecast_values))
        actual_values = actual_values[-min_len:]
        forecast_values = forecast_values[:min_len]
    
    rmse_ets = calculate_rmse(actual_values, forecast_values)
    
    aic = model.aic
    bic = model.bic
    
    print(f"En İyi Parametreler: {model.model.params}")
    print(f"RMSE: {rmse_ets}")
    print(f"AIC: {aic}")
    print(f"BIC: {bic}")
    
    return forecast_series, forecast_index

forecast_series, forecast_index = ets_forecast(data, 'Satıs_Tarihi', 'Adet')

plt.figure(figsize=(10, 7))
plt.plot(hafta.index, hafta['Adet'], label='Gerçek Veriler', color='royalblue')
plt.plot(forecast_index, forecast_series, label='Tahminler (ETS)', color='darkorange')
plt.fill_between(forecast_index, 
                 forecast_series - 1.96 * forecast_series.std(),
                 forecast_series + 1.96 * forecast_series.std(),
                 color='darkorange', alpha=0.2)

plt.title('Haftalık Satışlar ve Tahminler (ETS)')
plt.xlabel('Tarih')
plt.ylabel('Satışlar')
plt.legend()
plt.grid(True)
plt.xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2024-12-31'))
plt.show()
```
RMSE: 22.59259100408932
AIC: 1347.0405063216945
BIC: 1537.5535586836886
![image](https://github.com/user-attachments/assets/a76ef23e-5669-49f5-b9d0-58c84153377e)

ETS modelimizi yaptığımızda hatası ve AIC değeri auto arimaya göre daha iyi sonuç veriyor fakat grafiğe baktığımızda bazen gerçek verilerle tahminlerin eşleşmediğini görüyoruz. Ancak tahminlerimiz arimaya göre biraz daha iyi. İsteğe göre yapılcak işe göre model seçimi kişiye ve işe göre değişebilir.

### Haftalık Tahminler Görselleştirme 2024
```
model_haftalık = pm.ARIMA(
    order=(0, 0, 0),
    seasonal_order=(1, 1, 0, 52),
    maxiter=50,
    method='lbfgs',
    out_of_sample_size=0,
    scoring='mse',
    suppress_warnings=True,
    trend=None,
    with_intercept=False
)

model_haftalık.fit(hafta)
joblib.dump(model_haftalık, 'model_haftalık.pkl')
n_periods = 52  
forecast, conf_int = model_haftalık.predict(n_periods=n_periods, return_conf_int=True)
forecast_index = pd.date_range(start='2024-01-01', periods=n_periods, freq='W')

plt.figure(figsize=(14, 7))
plt.plot(hafta.index, hafta['Adet'], label='Gerçek Veriler', color='blue')
plt.plot(forecast_index, forecast, label='Tahminler (ARIMA)', color='orange')
plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='orange', alpha=0.3)

plt.title('Haftalık Satışlar ve 2024 Tahminleri (ARIMA)')
plt.xlabel('Tarih')
plt.ylabel('Satışlar')
plt.legend()
plt.grid(True)
plt.show()
```
![output_50_0](https://github.com/user-attachments/assets/d253ef6d-5846-4ff2-a307-58699367646e)
Burda 2024 için tahminlerimizi aldık. Gayet güzel olduğunu söyleyebiliriz.

```
model_haftalık_ets = ExponentialSmoothing(
    hafta['Adet'], 
    trend='add', 
    seasonal='mul', 
    seasonal_periods=52, 
    damped_trend=True
).fit()

# Modeli kaydetme
joblib.dump(model_haftalık_ets, 'model_haftalık_ets.pkl')

# 52 haftalık tahmin
n_periods = 52  
forecast_ets = model_haftalık_ets.forecast(n_periods)
forecast_index = pd.date_range(start='2024-01-01', periods=n_periods, freq='W')

# Grafikte gösterme
plt.figure(figsize=(14, 7))
plt.plot(hafta.index, hafta['Adet'], label='Gerçek Veriler', color='blue')
plt.plot(forecast_index, forecast_ets, label='Tahminler (ETS)', color='orange')
plt.fill_between(forecast_index, 
                 forecast_ets - 1.96 * forecast_ets.std(), 
                 forecast_ets + 1.96 * forecast_ets.std(), 
                 color='orange', alpha=0.3)

plt.title('Haftalık Satışlar ve 2024 Tahminleri (ETS)')
plt.xlabel('Tarih')
plt.ylabel('Satışlar')
plt.legend()
plt.grid(True)
plt.show()
```
![output_50_0](https://github.com/user-attachments/assets/90467ad5-a7ed-4729-a525-c241a3a04c97)
Burda 2024 için tahminlerimizi aldık. Gayet güzel olduğunu söyleyebiliriz.

### Aylık Tahminler Görselleştirme 2024
2024 için aylık olarak eğitip görselleştirdik.
```
hafta_aylık = hafta.resample('M').mean()

model_aylık = pm.ARIMA(
    order=(0, 0, 0),
    seasonal_order=(1, 1, 0, 12),  
    maxiter=50,
    method='lbfgs',
    out_of_sample_size=0,
    scoring='mse',
    suppress_warnings=True,
    trend=None,
    with_intercept=False
)

model_aylık.fit(hafta_aylık)
joblib.dump(model_aylık, 'model_aylık.pkl')
n_periods = 12  
forecast, conf_int = model_aylık.predict(n_periods=n_periods, return_conf_int=True)
forecast_index = pd.date_range(start='2024-01-01', periods=n_periods, freq='M')

combined = pd.concat([hafta_aylık, pd.Series(forecast, index=forecast_index)], axis=1)
combined.columns = ['Gerçek Veriler', 'Tahminler']
combined['Tahminler'] = combined['Tahminler'].fillna(method='ffill')

plt.figure(figsize=(14, 7))
plt.plot(combined.index, combined['Gerçek Veriler'], label='Gerçek Veriler', color='blue')
plt.plot(combined.index, combined['Tahminler'], label='Tahminler (ARIMA)', color='orange')
plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='orange', alpha=0.3)

plt.title('Aylık Satışlar ve 2024 Tahminleri (ARIMA)')
plt.xlabel('Tarih')
plt.ylabel('Satışlar')
plt.legend()
plt.grid(True)
plt.xlim(combined.index.min(), combined.index.max()) 
plt.show()
```
![output_53_0](https://github.com/user-attachments/assets/1781370d-e857-4073-ba74-e934934a2605)

```
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
import matplotlib.pyplot as plt
import pandas as pd

hafta_aylık = hafta.resample('M').mean()

model_aylık_ets = ExponentialSmoothing(
    hafta_aylık, 
    trend='add', 
    seasonal='mul', 
    seasonal_periods=12,
    damped_trend=True
).fit()

joblib.dump(model_aylık_ets, 'model_aylık_ets.pkl')

n_periods = 12  
forecast_ets = model_aylık_ets.forecast(n_periods)
forecast_index = pd.date_range(start='2024-01-01', periods=n_periods, freq='M')

combined = pd.concat([hafta_aylık, pd.Series(forecast_ets, index=forecast_index)], axis=1)
combined.columns = ['Gerçek Veriler', 'Tahminler']
combined['Tahminler'] = combined['Tahminler'].fillna(method='ffill')


plt.figure(figsize=(14, 7))
plt.plot(combined.index, combined['Gerçek Veriler'], label='Gerçek Veriler', color='blue')
plt.plot(combined.index, combined['Tahminler'], label='Tahminler (ETS)', color='orange')
plt.fill_between(forecast_index, 
                 forecast_ets - 1.96 * forecast_ets.std(), 
                 forecast_ets + 1.96 * forecast_ets.std(), 
                 color='orange', alpha=0.3)

plt.title('Aylık Satışlar ve 2024 Tahminleri (ETS)')
plt.xlabel('Tarih')
plt.ylabel('Satışlar')
plt.legend()
plt.grid(True)
plt.xlim(combined.index.min(), combined.index.max()) 
plt.show()
```
![output_53_0](https://github.com/user-attachments/assets/104d2214-997b-414a-a783-4065dd8f9b51)
Aylık görselleştirmelerimizide yaptık. Ben ben Tahminlerimi auto arima üzerinden yapacağım çünkü grafiğe daha çok uyuyor aralarındaki hata az olduğu için bu benim için kabul edilebilir.

## Modellerimiz Postman ile Deneyeceğiz
```
app = Flask(__name__)

model_haftalık = joblib.load('model_haftalık.pkl')
model_aylık = joblib.load('model_aylık.pkl')

def calculate_forecast(data, time_scale):
    if time_scale == 'haftalık':
        model = model_haftalık
        n_periods = 52
        freq = 'W'
    elif time_scale == 'aylık':
        data = data.resample('M').mean()
        model = model_aylık
        n_periods = 12
        freq = 'M'
    else:
        return None, None, None

    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    forecast_index = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=n_periods, freq=freq)
    return forecast, conf_int, forecast_index

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        time_scale = data.get('time_scale', 'aylık')
        tarih = pd.to_datetime(data['tarih'])
        adet = np.array(data['adet'])
        
        hafta = pd.DataFrame({'Adet': adet}, index=tarih)
        
        forecast, conf_int, forecast_index = calculate_forecast(hafta, time_scale)
        
        if forecast is None:
            return jsonify({'error': 'Invalid time scale provided.'}), 400

        result = {
            'Tahmin': forecast.tolist(),
            'Guven_aralıgı': conf_int.tolist(),
            'Tarihler': forecast_index.strftime('%Y-%m-%d').tolist()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, port=9874)
```
#### Input
{
    "time_scale": "aylık",
    "tarih": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"],
    "adet": [30, 40, 35, 50, 60, 45]
}
#### Output
{
    "Guven_aralıgı": [
        [
            37.432102486294745,
            78.58035128456567
        ],
        [
            27.852757308506256,
            69.0010061067772
        ],
        [
            27.987457219332246,
            69.13570601760318
        ],
        [
            27.689208441359142,
            68.83745723963008
        ],
        [
            31.092217618389217,
            72.24046641666015
        ],
        [
            29.63865138905181,
            70.78690018732274
        ],
        [
            37.199793868045745,
            78.34804266631669
        ],
        [
            24.346752076756662,
            65.4950008750276
        ],
        [
            35.493862734230305,
            76.64211153250125
        ],
        [
            35.83704184376787,
            76.98529064203882
        ],
        [
            19.028364420979845,
            60.17661321925078
        ],
        [
            44.28641867858552,
            85.43466747685646
        ]
    ],
    "Tahmin": [
        58.00622688543021,
        48.426881707641726,
        48.56158161846771,
        48.26333284049461,
        51.666342017524684,
        50.21277578818728,
        57.773918267181216,
        44.92087647589213,
        56.067987133365776,
        56.41116624290334,
        39.60248882011531,
        64.86054307772099
    ],
    "Tarihler": [
        "2023-07-31",
        "2023-08-31",
        "2023-09-30",
        "2023-10-31",
        "2023-11-30",
        "2023-12-31",
        "2024-01-31",
        "2024-02-29",
        "2024-03-31",
        "2024-04-30",
        "2024-05-31",
        "2024-06-30"
    ]
}
Uygulamamız çalıştı. Verilen input verileri için model, 2023 Temmuz'dan başlayarak 2024 Haziran'a kadar tahminler üretmiş. Tahminler, genellikle hem artış hem de düşüş gösteriyor, bu da modelin belirli dönemlerde satışların arttığını ve bazı dönemlerde azaldığını öngördüğünü gösteriyor.
