import pandas as pd
import argparse
import os

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--max_workers', default=20)
parser.add_argument('--data')
parser.add_argument('--result', default='')

args = parser.parse_args()


# 读取数据
data_path = args.data
texts = pd.read_csv(data_path)
num_texts = len(texts)
cols = list(texts.columns) + ['perspective_api_results']
del texts


# 访问API的函数体
from googleapiclient import discovery
import httplib2

def perspective_analyze(text, proxy=False, proxy_host="192.168.50.183", proxy_port=7890):

    API_KEY = 'AIzaSyAPlZbUYD2pYxmz_CsDPUe5vNIQFiGlxS0'

    proxy_info = httplib2.ProxyInfo(
        proxy_type=httplib2.socks.PROXY_TYPE_HTTP,
        proxy_host=proxy_host,
        proxy_port=proxy_port
    )

    http = httplib2.Http(
        timeout=10,
        proxy_info=proxy_info,
        disable_ssl_certificate_validation=False
    )

    client = discovery.build(
        serviceName="commentanalyzer",
        version="v1alpha1",
        http=http if proxy == True else None,
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        'comment': {'text': text},
        'requestedAttributes': {
            'TOXICITY': {}
        }
    }
    
    return client.comments().analyze(body=analyze_request).execute()

def toxicity_analyze(x):
    try:
        return perspective_analyze(x)
    except:
        return None

from concurrent.futures import ThreadPoolExecutor
def parallel_analyze(data, func, n_workers=4):
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(func, data))
    return results
    

# 开始分析
from tqdm import tqdm
from math import ceil

result_path = 'output/toxicity_of_' + os.path.basename(data_path) if args.result == '' else args.result
chunk_size = 2500   # 分批处理，每批的数量
n_workers = int(args.max_workers)   # 线程数量

print('\n')
print(f"data_path: {data_path}")
print(f"result_path: {result_path}")
print(f"number of texts: {num_texts}")

df_tmp = pd.DataFrame(columns=cols)
df_tmp.to_csv(result_path, header=True, index=False)

for chunk in tqdm(pd.read_csv(data_path, chunksize=chunk_size), total=ceil(num_texts / chunk_size)):
    tmp = parallel_analyze(list(chunk['text']), toxicity_analyze, n_workers=n_workers)
    chunk.loc[:, 'perspective_api_results'] = tmp
    chunk.to_csv(result_path, index=False, mode='a', header=False)

print("Done. \n\n\n")