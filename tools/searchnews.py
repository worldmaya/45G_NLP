import requests
from bs4 import BeautifulSoup
def get_score(i,j):
    count = 0
    for char_j in j:
        for char_i in i:
           if char_i==char_j:
               count+=1
               break
    score = (count/len(j)+count/len(i))/2
    return score
def ishot(sentence,url):
    resp = requests.get(url = url)
    resp.encoding='utf-8'
    html = resp.text
    soup = BeautifulSoup(html,"html.parser")
    contents = []
    try:
        for i in  range(100):
            a = soup.find_all('p')[i].string
            contents.append(a)
    except:
        pass
    score = []
    for i in contents:
        score.append(get_score(str(i),sentence))
    return max(score)

def calculate_count_score(sentence,url):
    score = ishot(sentence, url)
    return score

if __name__ == '__main__':
    sentence = '海军轰炸机南海演练'
    url = 'https://mil.news.sina.com.cn/'
    print(ishot(sentence,url))