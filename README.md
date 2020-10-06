# App-analysation

## 簡介
針對Play Store上的App相關資訊，進行資料對照及分析，<br>
根據分析結果篩選需要的特徵<br>
使用SVC、DecisionTree、RamdomForest、LightGBM、XGBoost等機器學習算法<br>
推測App是否熱門(評價>4為熱門)。
<br>
<br>

## 資料集
Mobile App Store(7200 apps)中的AppleStore.csv <br>
https://www.kaggle.com/ramamet4/app-store-apple-data-set-10k-apps
<br>
<br>

## 使用方法
將AppleStore.csv放置系統當前目錄<br>
執行appSuccess.py
<br>
<br>

## 資料集視覺化
#### App各種類數量<br>
![](https://github.com/sha310139/App-analysation/blob/main/genre.png)  
<br>
#### App各種類之user rating<br>
![](https://github.com/sha310139/App-analysation/blob/main/user_rate.png)  
<br>
<br>

## 資料集預處理
* size_bytes->size_bytes_in_MB：將App大小從Bytes為單位轉成用MB為單位
* price->isNotfree：將價格二分化成免費與付費App
* rating_count_tot-rating_count_ver ->rating_count_before：所有版本評分的用戶數量減去只有評最新版本的用戶數量
<br>
<br>

## 相關係數分析
下圖為label相關係數
多數label與user_rating相關係數不高
以ipadSc_urls.num較為高(0.265)
![](https://github.com/sha310139/App-analysation/blob/main/matrix.png)  
<br>
<br>


## 標記數據集
將數據集標上label，以user_rating作為target。<br>
該app平均用戶評價>4的數據為熱門(高評價)的App，標記為1<br>
該app平均用戶評價<=4的數據為較不熱門(低評價)的App，標記為0<br>
下圖為成功與失敗的App數量<br>
![](https://github.com/sha310139/App-analysation/blob/main/count.png)  
<br>
<br>

## 選擇特徵集
依據各個特徵與user_rating的相關係數選擇特徵集，大部分相關係數都不高，因此將所有相關係數為正的特徵加入。<br>
* size_bytes_in_MB(0.066)
* isNotFree(0.112)
* Price(0.046)
* rating_count_before(0.080)
* ipadSc_urls.num(0.265)
* lang.num(0.171)
* vpp_lic(0.069)
* prime_genre(非數值無法得知相關係數, 但或許與評價高低有關，因此一起加入)
<br>
<br>

## 切分訓練集、測試集
    train_test_split(df_train.values, target, test_size=0.2, random_state=1989, stratify=target)
  * 測試集大小 : 20%
  * ramdom_state=1989 : 偽隨機中設固定一整數, 藉此產出相同結果
  * Stratify : 按照target的比例分資料, 避免某個set中全是成功App或全是失敗App
<br>
<br>
  
## 結果
使用KFold cross validate<br>
n_splits=10<br>
    Classfier_name	  train_score	test_score
    
    SVC	          0.949655	0.623874
    DecisionTree	  0.682507	0.665278
    RamdomForest	  0.693869	0.68112
    LGBM	          0.770228	0.681811
    XGB	          0.93539	0.671248
<br>
<br>
