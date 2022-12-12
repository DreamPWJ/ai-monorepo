import os
import re
import time

import requests

"""
  @author 潘维吉
  @date 2022/12/05 13:22
  @email 406798106@qq.com
  @description Python爬虫
"""


def imgdata_set(save_path, word, epoch):
    q = 0  # 停止爬取图片条件
    a = 0  # 图片名称
    while (True):
        time.sleep(1)
        url = "https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word={}&pn={}&ct=&ic=0&lm=-1&width=0&height=0".format(
            word, q)
        # word=需要搜索的名字
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.56'
        }
        response = requests.get(url, headers=headers)
        # print(response.request.headers)
        html = response.text
        # print(html)
        urls = re.findall('"objURL":"(.*?)"', html)
        # print(urls)
        for url in urls:
            print(a)  # 图片的名字
            response = requests.get(url, headers=headers)
            image = response.content
            with open(os.path.join(save_path, "{}.jpg".format(a)), 'wb') as f:
                f.write(image)
            a = a + 1
        q = q + 20
        if (q / 20) >= int(epoch):
            break


if __name__ == "__main__":
    save_path = input('你想保存的路径: ')  # 确保目录已存在
    word = input('你想要下载什么图片？请输入: ')
    epoch = input('你想要下载几轮图片？请输入(每轮默认为60张左右图片): ')  # 需要执行几轮
    imgdata_set(save_path, word, epoch)
