import re
import requests
import os
from threading import Thread
import json
import time

max_thread = 16
now_thread = 0


headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.8',
    'Cache-Control': 'max-age=0',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/48.0.2564.116 Safari/537.36',
    'Connection': 'keep-alive',
    'Referer': 'http://www.baidu.com/'
}





def get_file(company_name:str, date_filed, CIK:str, addres:str):
    global now_thread
    dic = dict()
    path = './files/'+CIK+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    address = "https://www.sec.gov/Archives/"+addres
    try:
        filename = address.split('/')[-1]
        pa = path + filename
        a = requests.get(address, headers=headers)
        with open(pa, 'wb') as f:
            f.write(a.content)
        pa_j = path+filename+'.json'
        dic['CIK'] = CIK
        dic['company_name'] = company_name
        dic['date_filed'] = date_filed
        with open(pa_j, 'w') as f:
            json.dump(dic, f)
    except Exception as e:
        print(e)
    # time.sleep(0.1)
    now_thread -= 1

# get_file('https://www.sec.gov/Archives/edgar/data/1000264/0001193125-22-260148.txt')
def company():
    global now_thread, max_thread
    with open('./company.idx', 'r') as f:
        lines=f.readlines()[10:]
        for i in range(10000):
            while now_thread >= max_thread:
                time.sleep(0.1)
            time.sleep(0.1)
            # while now_thread >= max_thread:
            #     time.sleep(0.1)
            company_name = lines[i][:62].strip()
            form_type = lines[i][62:74].strip()
            CIK = lines[i][74:86].strip()
            date_filed = lines[i][86:98].strip()
            part_url = lines[i][98:].strip()
            now_thread += 1
            Thread(target=get_file, args=(company_name, date_filed, CIK, part_url)).start()
            # get_file(CIK, part_url)
            if i%100 == 0:
                print(i,now_thread)

company()