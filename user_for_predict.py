# 撈取昨天有上線且7天內沒領過推薦禮包的免費玩家，以方便後續用 model.py 進行 商品推薦(predict)
import urllib
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO
import os
import sys

# 商品很多會先進行商品分類，方便後續匹配
p = os.path.join("item_category.csv")
category = pd.read_csv(p)
category = category.set_index('item')['category'].to_dict()  #索引為key，對應item_category欄位

#%% config
# ===== config =====
def config_read(file_name):
    # Read config file
    if sys.platform.startswith('linux'):
        file_path = r'linux 設定檔路徑'  # linux文件路径
    elif sys.platform.startswith('win'):
        file_path = r'Windows 設定檔路徑'
    else:
        print("無法判斷程式執行的作業系統")

    file_path = os.path.join(file_path, file_name) #完整設定檔路徑
    #讀入json
    with open(file_path, 'r') as file:
        config = json.load(file)
    
    config = pd.json_normalize(config).stack().reset_index(level=0, drop=True) #刪除多的一層索引
    config = config.to_dict()
    return config

config = config_read(file_name = 'all_setting.json')
API_config = config_read(file_name = 'API_setting.json')
token = API_config['game.token']


#%%
class get_data(object):
    
    #被呼叫時一定會先執行
    def __init__(self):
        pass

    #抓出24小時內的上線用戶清單，確認有派發獎勵，這邊後續可以考慮改成有提供獎勵的用戶埋點
    @staticmethod
    def dau_24hr(start_time, end_time):  
        #抓近24小時有上線的用戶，且只抓註冊7天以上
        SQL ='''
        --- "$part_date"
        SELECT a."#account_id" , a.login_date
        from
        (
            select "#account_id"
            , date(min("#event_time")) as login_date
            from game_data
            where ("#event_time" between cast('%s' as timestamp) and cast('%s' as timestamp)) 
            AND ("$part_event" IN ('login'))
            GROUP BY "#account_id"
        ) a
        join ta.v_user_18 b on a."#account_id" = b."#account_id" and date_diff('day', date(b."register_time"), a.login_date) >= 7
        '''%(start_time, end_time)
        r = requests.post(url = 'http://API_address/querySql?token='+token,  
                          headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                          data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header')
        s=str(r.content,'utf-8')
        data = pd.read_csv(StringIO(s))
        data = data.set_index('#account_id')['login_date'].to_dict()
        return data

    
    #讀入已經預測的id list歷史清單，後續預測將排除此份清單
    @staticmethod
    def update_id_list(dau_):
        
        with open('', 'r') as infile:
            data = json.load(infile)
            data = {int(k): v for k, v in data.items()}     #把Json讀出來的鍵值轉為int
        #若是空值或開始日期至今天<=7天則會保留，也就是刪除7天以上的清單
        data = {k: v for k, v in data.items() if bool(v) == False or (datetime.now()-datetime.strptime(v, '%Y-%m-%d')).days <= 7}  
        #更新有上線的人到空值清單中，若元素同時出現在已預測list及昨日上線清單，且預測list是空值則更新
        data = {k: dau_[k] if k in dau_ and k in data and bool(v) == False else v for k, v in data.items()}
        return data 
    
    #抓出24小時內的上線用戶清單，排除付費玩家
    @staticmethod
    def dau_free(dau_):
        #抓出歷史付費用戶
        SQL ='''
        select distinct "#account_id"
        from game_data
        where ("$part_date" >='2020-01-01') AND ("$part_event" IN ('order_finish'))
        '''
        r = requests.post(url = 'http://API_address/querySql?token='+token,  
                          headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                          data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        data =  pd.read_csv(StringIO(s))
        pay = list(data['#account_id'].values)  #將付費用戶ID以list記錄下來
        #x = date #pd.to_datetime("today").strftime("%Y-%m-%d")
        dau_free = [x for x in dau_ if x not in pay]   #排除在pay中的用戶
        return dau_free
    
    
    @staticmethod
    def find_count(x):
        '''抓出商品數量'''
        if "*" not in x:           #item_gain有些是空值
            return 1
        else:                      #有值的，抓取*後面的數量
            return int(x.split('*')[1])
        
    @staticmethod
    def item(x):
        '''將途徑、名稱連接在一起'''
        if x[1]=='':
            return x[0]
        else:
            return x[0]+'_'+x[1]
    
    @staticmethod
    def clothes(x):
        '''由於時裝太多名稱，故將名稱對應回大的分類，後續建模採用這分類做為名稱'''
        if '购买服装' in x:
            return category[x]
        else:
            return x
  
    
    #抓出預測清單的鑽石購買商品
    @staticmethod
    def diamond_item(predict):
        #讀入PM提供的綁鑽商品的分類表
        SQL ='''
        select "#account_id",reason,item_gain
        from game_data
        where ("$part_date" >='2020-01-01') AND ("$part_event" IN ('diamond_consume')) and diamond_consume_amount=0
        '''
        r = requests.post(url = 'http://API_address/querySql?token='+token,  
                              headers ={'Content-Type': 'application/x-www-form-urlencoded' },
                              data= 'sql='+ urllib.parse.quote(SQL) + '&format=csv_header') 
        s=str(r.content,'utf-8')
        data = pd.read_csv(StringIO(s))
        data = data[data['#account_id'].isin(list(predict['#account_id']))]  #只篩選要被預測的清單
        data['item_gain'].fillna('',inplace = True)
        data['item_gain'] = data['item_gain'].apply(lambda x:str(x).split("\t")) #將多商品資料用"\t"分割，傳回list
        data = data.explode("item_gain") #將list解開巢狀結構，變成列
        data['count'] = data['item_gain'].apply(lambda x : get_data.find_count(x))   #取出商品數量
        data['item_gain'] = data['item_gain'].apply(lambda x:x.split('*')[0])  #取出商品名稱
        data['item'] = data[['reason',"item_gain"]].apply(lambda x:get_data.item(x),axis=1)  #連結途徑+商品名稱，代表1件商品
        data = data[data['item'].isin(list(category.keys()))]    #限定商品只能在'item_category.csv'中，其他的難以推薦
        data['item'] = data['item'].apply(lambda x:get_data.clothes(x))   #把時裝名稱替換，改為時裝大的歸類 (來自excel)
        
        #建立crosstable
        data = pd.crosstab(data['#account_id'],data['item'],values=data['count'],aggfunc=sum) #將直的內容展開
        data.fillna(0,inplace=True)
        return data
    

class predict_dataset(object):
   
    def __init__(self, save, mode, **kwargs):   #remove date (self,path,mode,*days)
        self.mode = mode
        self.save = save
        
        #如果kwargs中包含"date"和"path"這兩個鍵，則把它們的值賦給對應的屬性self.date和self.path；否則這兩個屬性的值將是None
        for k in kwargs.keys():
            if k in ["path","date"]:
                self.__setattr__(k, kwargs[k])

    def build(self):
        
        '''
        這段為了處理7天內已經領過獎勵玩家，不會再預測
        讀入目前的預測ID清單檔案，因為第一次執行時會沒檔案，故用try except處理
        id_list:歷史預測的ID清單:有領到推薦的日期
        dau_昨天有上線的清單
        '''
        try:
            start_time = datetime.now() - timedelta(hours=24)    #時間區間抓當前時間前24小時內有上線的用戶
            end_time = datetime.now()
            dau_ = get_data.dau_24hr(start_time, end_time)   #24小時內有上線清單
            exclude_id = get_data.update_id_list(dau_)             #要排除預測的id list
            dau_ = [x for x in dau_ if x not in exclude_id]   #上線清單排除exclude_id，作為今天要預測的新清單
            with open('', 'w') as outfile:    #紀錄每個ID及領到推薦日期，覆蓋模式
                json.dump(exclude_id, outfile)
        except:
            pass
        
        dau_ = get_data.dau_free(dau_)    #排除過付費、領獎中的玩家的最終預測清單
        #抓取當日DAU (排除付費玩家)
        predict = pd.DataFrame([])
        predict['#account_id']= dau_
        predict = predict.drop_duplicates()    #先抓出要預測的ID list
        
        if self.save:
            predict.to_csv(self.path, index = False) 
        return predict
        
