import os
import re
import time
import numpy as np
import unicodedata

import html2text
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional


class SplitFile:
    text_maker = html2text.HTML2Text()
    text_maker.BYPASS_TABLES = False
    text_maker.IGNORE_TABLES = False
    text_maker.RE_MD_CHARS_MATCHER_ALL = True
    # 标准序列编号
    rc = ['1', '1A', '1B', '2', '3', '4', '4A', '5', '6', '7', '7A', '8', '9', '9A', '9B', '10', '11', '12', '13', '14', '15']
    # 源数据根目录
    base_dir: str
    # 切分文本后根目录
    out_dir: str
    # 仅处理简化版的
    simple_files: List[str]
    # 已切分数组
    history: set
    # 已切分文件清单
    history_log: str
    # 异常文件清单
    error_log: str
    error: set
    # TF-IDF矩阵
    rs = ()

    def __init__(self, base_dir, out_dir, corpus_dir, history_log, simple_log):
        self.base_dir = base_dir
        self.out_dir = out_dir
        self.corpus_dir = corpus_dir
        self.history_log = history_log
        self.simple_files = self.load_txt_lines(simple_log)
        self.history = set(self.load_txt_lines(self.history_log))
        self.error_log = '../err.log'
        self.error = set(self.load_txt_lines(self.error_log))
        self.init_tfidf_matrix()
        print()

    def init_tfidf_matrix(self):
        corpus = []
        for r in self.rc:
            with open(self.corpus_dir + os.sep + r + '.txt', 'r', encoding='UTF-8') as f:
                corpus.append(f.read())
        self.rs = self.calc_tfidf_matrix(corpus)

    # 开始解析所有txt文件
    def walk(self):
        # 获取根目录，递归得到所以txt文件的路径
        list_dirs = os.walk(self.base_dir)
        # 计数器
        turn = 0
        for dirpath, dirnames, filenames in list_dirs:
            for filename in filenames:
                if filename[-4:] == '.txt':
                    turn += 1
                    
                    abspath = dirpath + os.sep + filename
                    if filename in self.history or abspath in self.error:
                        continue
                    try:
                        print(f"turn:{turn}")
                        self.simple_files.index(filename)
                        self.get_10k_body(abspath)
                    except Exception as err:
                        print('walk:',err)
        print('total files:', turn)

    # 加载txt文件，按行读取
    @staticmethod
    def load_txt_lines(file_path):
        lines = []
        try:
            with open(file_path) as file:
                fLines = file.readlines()
            for line in fLines:
                line = line.strip('\n')
                if line:
                    lines.append(line.split('\\')[-1])
        except Exception as err:
            print(err)
        return lines

    # 提取10-K内容部分
    def get_10k_body(self, abspath):
        # 行标记：2表示<TYPE>10-K开始；3表示存在<FILENAME>；5表示<TEXT>开始；7表示<TEXT>结束
        flag = 1
        # 行号
        line_no = 0
        # print(abspath)
        file = open(abspath)
        # 正文，按字符串存储，存在换行符，即\r\n
        content = ''
        # 正文，按行存储
        lines = []
        # while 1:
        fileLines = file.readlines()
        for line in fileLines:
            line_no = line_no + 1
            line = self.clean_non_standard_chars(line)
            if line == '<TYPE>10-K':
                flag *= 2
            if line.startswith('<FILENAME>'):
                flag *= 3
            if line == '<TEXT>':
                flag *= 5
                continue
            if line == '</TEXT>':
                flag = 7
                break
            if flag % 5 == 0:
                content = content + line + '\n'
                lines.append(line)
        file.close()

        file_type = self.get_doc_type(content)
        if file_type == 3:
            self.match_item_from_lines(abspath, lines, self.rs)
        if file_type == 2:
            new_lines = self.get_lines_from_content(content)
            self.match_item_from_lines(abspath, new_lines, self.rs)
        self.logger('HISTORY', abspath.split(os.sep)[-1])

    # 日志记录
    def logger(self, log_type, line):
        if log_type == 'ERROR':
            file_path = self.error_log
        elif log_type == 'HISTORY':
            file_path = self.history_log
        with open(file_path, "a+", encoding='UTF-8') as file:
            file.write(line + '\n')
            file.close()

    # 计算相似度，返回值（法一，法二）
    def calc_similarity(self, item_text, feature_names, tfidf, ts):
        if item_text is None:
            return np.zeros(21).tolist()
        L = item_text.split(' ')
        W = []
        # 与标准21个Document的相似度
        RS = []
        RS5 = []
        # 测试文本分词后，获取在TF-IDF矩阵中的权重
        WS = np.zeros((21, len(feature_names)))
        WS5 = np.zeros((21, len(feature_names)))
        for f in L:
            try:
                W.append(feature_names.index(f.lower()))
            except Exception as e:
                # print(str(e))
                W.append(-1)
        for j in range(20):
            for i in W:
                if i > -1:
                    WS[j][i] = tfidf[j][i]
                    WS5[j][i] = 0.5
            JS = np.sum(np.square(WS[j]))
            JS5 = np.sum(np.multiply(WS[j], 0.5))
            HJS5 = np.sum(np.square(WS5[j]))
            rs = 0 if JS == 0 else JS / (ts[j] * np.sqrt(JS))
            rs5 = 0 if JS5 == 0 else JS5 / (ts[j] * np.sqrt(HJS5))
            RS.append(rs)
            RS5.append(rs5)
        return RS5

    # 返回值（最高值，最高值行列索引）
    def get_max_val(self, item_text, feature_names, tfidf, ts):
        similarity = self.calc_similarity(item_text, feature_names, tfidf, ts)
        max_value = max(similarity)
        return similarity.index(max_value), max_value

    # 获取“item”编号索引，均为标准格式，如：Item 1/Item 1A
    def get_standard_num(self, item):
        try:
            searched = re.search(r'\d+.*', item, re.I)
            num = searched.group()
            return self.rc.index(num)
        except:
            return None

    # 判断文档内容类型，2表示html；3表示txt；5表示xbrl
    def get_doc_type(self, content):
        if content.find('<XBRL') > 0:
            return 5
        cleaned_text = self.clean_non_standard_tags(content)
        beautiful_soup = BeautifulSoup(cleaned_text, 'lxml')
        div_elem = beautiful_soup.find_all('div')
        p_elem = beautiful_soup.find_all('p')
        if len(div_elem) >= 5 or len(p_elem) >= 5:
            return 2
        return 3

    # 处理HTML，先格式化，再做文本转化（为保证与预览的效果一致）
    def get_lines_from_content(self, content):
        cleaned_text = self.clean_non_standard_tags(content)
        # 去除html标签，方法一
        text = self.text_maker.handle(cleaned_text)
        # 去除html标签，方法二
        # beautiful_soup = BeautifulSoup(cleaned_text, 'lxml')
        # 格式化HTML   prettify
        # prettify = beautiful_soup.get_text()
        # text = re.sub('<[^<]+?>', '', prettify, flags=re.I)
        # 去除html标签，方法三
        # text = nltk.clean_html(cleaned_text)
        # 去除html标签，方法四
        # text = re.sub('<[^<]+?>', '', cleaned_text, flags=re.I)
        return text.split('\n')

    # 开始匹配Item
    def match_item_from_lines(self, abspath, lines, rs):
        line_no = 1
        # 二维数组
        long_series = []
        for line in lines:
            if len(line) > 0:
                strip_line = self.clean_non_standard_tags(line.strip())
                matched = re.match(r'^ITEM[S]?\s*[0-9]{1,2}\(?[ABCDE]?\)?', strip_line, re.I)
                #  and len(pattern.findall(strip_line)) == 1
                if matched:
                    # TODO 未处理9 A这种情况与 9 AND 冲突了
                    maybe_item = self.filter_special_item(strip_line)
                    # maybe_item = matched.group()
                    item_val = strip_line[len(maybe_item):]
                    # 计算Item的文本相似度
                    maybe_idx = self.get_max_val(item_val, rs[0], rs[1], rs[2])
                    curr_idx = self.get_standard_num(maybe_item)
                    # 非法编码，也就是没有在rc数组内的，比如：15(1)
                    if curr_idx is None:
                        continue
                    if 0 < maybe_idx[1] < 1:
                        continue
                    long_series.append([curr_idx, line_no])
            line_no += 1
        matched_series = self.get_matched_series(long_series)
        if matched_series is not None:
            self.split_item_segm(lines, matched_series, abspath)
        else:
            self.logger('ERROR', abspath.split(os.sep)[-1])
        return None

    # 将Item 1和Item 7文本写入txt
    def split_item_segm(self, lines, matched_series, abspath):
        tags = abspath.split(os.sep)
        out_path = self.out_dir + os.sep + os.path.join(tags[-4],  tags[-3],  tags[-2]) + os.sep
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # 开始切分
        for i in range(len(matched_series)):
            if matched_series[i][0] == 0:
                if i == len(matched_series) - 1:
                    sub_lines = lines[matched_series[i][1]-1:]
                else:
                    sub_lines = lines[matched_series[i][1]-1:matched_series[i+1][1]-1]
                # Item 1
                with open(out_path + '1.txt', 'w', encoding='UTF-8') as f:
                    f.write('\n'.join(sub_lines))
            elif matched_series[i][0] == 9:
                if i == len(matched_series) - 1:
                    sub_lines = lines[matched_series[i][1]-1:]
                else:
                    sub_lines = lines[matched_series[i][1]-1:matched_series[i+1][1]-1]
                # Item 7
                with open(out_path + '7.txt', 'w', encoding='UTF-8') as f:
                    f.write('\n'.join(sub_lines))
            elif matched_series[i][0] == 10:
                if i == len(matched_series) - 1:
                    sub_lines = lines[matched_series[i][1]-1:]
                else:
                    sub_lines = lines[matched_series[i][1]-1:matched_series[i+1][1]-1]
                # Item 7
                with open(out_path + '7A.txt', 'w', encoding='UTF-8') as f:
                    f.write('\n'.join(sub_lines))

    # 利用差分算法，进行序列切割，再求最大方差
    def get_matched_series(self, long_series):
        if len(long_series) <= 2:
            return long_series
        diff = [long_series[i+2][0] - long_series[i][0] for i in range(len(long_series) - 2)]
        breakpoints = []
        for d in range(len(diff) - 1):
            if diff[d] < 0 and diff[d + 1] < 0:
                breakpoints.append(d + 2)
        breakpoints_len = len(breakpoints)
        if breakpoints_len == 0:
            return long_series
        if breakpoints_len == 1:
            return self.get_max_variance(long_series, breakpoints)
        return None

    # 切分并求方差
    def get_max_variance(self, long_series, breakpoints):
        if len(breakpoints) == 0:
            return long_series
        elif len(breakpoints) == 1:
            # 第一个
            first_raw = self.get_mean_var_series(long_series[:breakpoints[0]])
            first_mean = np.mean(first_raw)
            first_var = np.var(first_raw)
            # 第二个
            second_raw = self.get_mean_var_series(long_series[breakpoints[0]:])
            second_mean = np.mean(second_raw)
            second_var = np.var(second_raw)
            if first_mean < second_mean and first_var < second_var:
                return long_series[breakpoints[0]:]
            if first_mean > second_mean and first_var > second_var:
                return long_series[:breakpoints[0]]
            return None

    # 计算行距，返回数组，第一个默认行距为0
    @staticmethod
    def get_mean_var_series(single_series):
        line_nos = [0]
        for i in range(1, len(single_series)):
            line_nos.append(single_series[i][1] - single_series[i-1][1])
        return line_nos

    # item 9 A AND/item 9 A/item 9 AS/item 9 AND
    @staticmethod
    def filter_special_item(item):
        ori_no = ['1 A', '1 B', '4 A', '7 A', '9 A', '9 B', '1 a', '1 b', '4 a', '7 a', '9 a', '9 b']
        std_no = item
        for x in ori_no:
            idx = item.find(x)
            if idx > -1:
                if (idx == len(item) - 3) or item[idx + 3].isspace():
                    std_no = std_no.replace(x, x.replace(' ', ''))
        matched = re.match(r'^ITEM[S]?\s*[0-9]{1,2}\(?[ABCDE]?\)?', std_no, re.I)
        if matched:
            return matched.group()
        return None

    # 清除行内元素，如：font、s、b、a、del、strong、span
    @staticmethod
    def clean_non_standard_tags(content):
        # font
        norm_text = re.sub(r'<[\s]*?font[^>]*?>', '', content, flags=re.I)
        norm_text = re.sub(r'<[\s]*?\/[\s]*?font[\s]*?>', '', norm_text, flags=re.I)
        # s
        norm_text = re.sub(r'<[\s]*?s[^>]*?>', '', norm_text, flags=re.I)
        norm_text = re.sub(r'<[\s]*?\/[\s]*?s[\s]*?>', '', norm_text, flags=re.I)
        # b
        norm_text = re.sub(r'<[\s]*?b[^>]*?>', '', norm_text, flags=re.I)
        norm_text = re.sub(r'<[\s]*?\/[\s]*?b[\s]*?>', '', norm_text, flags=re.I)
        # a
        norm_text = re.sub(r'<[\s]*?a[^>]*?>', '', norm_text, flags=re.I)
        norm_text = re.sub(r'<[\s]*?\/[\s]*?a[\s]*?>', '', norm_text, flags=re.I)
        # del
        norm_text = re.sub(r'<[\s]*?del[^>]*?>', '', norm_text, flags=re.I)
        norm_text = re.sub(r'<[\s]*?\/[\s]*?del[\s]*?>', '', norm_text, flags=re.I)
        # strong
        norm_text = re.sub(r'<[\s]*?strong[^>]*?>', '', norm_text, flags=re.I)
        norm_text = re.sub(r'<[\s]*?\/[\s]*?strong[\s]*?>', '', norm_text, flags=re.I)
        # page
        norm_text = re.sub(r'<[\s]*?page[^>]*?>', '', norm_text, flags=re.I)
        norm_text = re.sub(r'<[\s]*?\/[\s]*?page[\s]*?>', '', norm_text, flags=re.I)
        # caption
        norm_text = re.sub(r'<[\s]*?caption[^>]*?>', '', norm_text, flags=re.I)
        norm_text = re.sub(r'<[\s]*?\/[\s]*?caption[\s]*?>', '', norm_text, flags=re.I)
        # fn
        norm_text = re.sub(r'<[\s]*?fn[^>]*?>', '', norm_text, flags=re.I)
        norm_text = re.sub(r'<[\s]*?\/[\s]*?fn[\s]*?>', '', norm_text, flags=re.I)
        # span
        norm_text = re.sub(r'<[\s]*?span[^>]*?>', '', norm_text, flags=re.I)
        norm_text = re.sub(r'<[\s]*?\/[\s]*?span[\s]*?>', '', norm_text, flags=re.I)
        # hr
        norm_text = re.sub(r'<[\s]*?hr[^>]*?>', '', norm_text, flags=re.I)
        norm_text = re.sub(r'<[\s]*?\/[\s]*?hr[\s]*?>', '', norm_text, flags=re.I)
        # u
        norm_text = re.sub(r'<[\s]*?u[^>]*?>', '', norm_text, flags=re.I)
        norm_text = re.sub(r'<[\s]*?\/[\s]*?u[\s]*?>', '', norm_text, flags=re.I)
        # pre
        norm_text = re.sub(r'<[\s]*?pre[^>]*?>', '', norm_text, flags=re.I)
        return re.sub(r'<[\s]*?\/[\s]*?pre[\s]*?>', '', norm_text, flags=re.I)

    # 清除源字符串（行）中的特殊字符
    @staticmethod
    def clean_non_standard_chars(line):
        norm_item = unicodedata.normalize('NFKC', line.strip())
        # 处理特殊字符(1)
        norm_item = re.sub(r'\t', ' ', norm_item, flags=re.I)
        norm_item = re.sub('\xa0', ' ', norm_item, flags=re.I)
        norm_item = re.sub('&nbsp;', ' ', norm_item, flags=re.I)
        norm_item = re.sub('&#160;', '', norm_item, flags=re.I)
        norm_item = re.sub('item l', 'item 1', norm_item, flags=re.I)
        # 处理特殊字符(2)，格式化HTML时会有问题
        # norm_item = re.sub(r'[!"#$%&()*+,:;=?@[\]^_`{|}~—ü‘’“”︹§ξ]', ' ', norm_item, flags=re.I)
        # 多个空格变为一个空格
        norm_item = norm_item.replace(r'\s+', '')
        # 英文数字转阿拉伯数字
        norm_item = re.sub(' one', ' 1', norm_item, flags=re.I)
        norm_item = re.sub(' two', ' 2', norm_item, flags=re.I)
        norm_item = re.sub(' three', ' 3', norm_item, flags=re.I)
        norm_item = re.sub(' fourteen', ' 14', norm_item, flags=re.I)
        norm_item = re.sub(' four', ' 4', norm_item, flags=re.I)
        norm_item = re.sub(' five', ' 5', norm_item, flags=re.I)
        norm_item = re.sub(' six', ' 6', norm_item, flags=re.I)
        norm_item = re.sub(' seven', ' 7', norm_item, flags=re.I)
        norm_item = re.sub(' eight', ' 8', norm_item, flags=re.I)
        norm_item = re.sub(' nine', ' 9', norm_item, flags=re.I)
        norm_item = re.sub(' ten', ' 10', norm_item, flags=re.I)
        norm_item = re.sub(' zero', ' 0', norm_item, flags=re.I)
        norm_item = re.sub(' eleven', ' 11', norm_item, flags=re.I)
        norm_item = re.sub(' twelve', ' 12', norm_item, flags=re.I)
        norm_item = re.sub(' thirteen', ' 13', norm_item, flags=re.I)
        norm_item = re.sub(' fifteen', ' 15', norm_item, flags=re.I)
        norm_item = re.sub(' sixteen', ' 16', norm_item, flags=re.I)
        # item后加个空格
        if re.match(r'^item\d+', norm_item, flags=re.I):
            norm_item = re.sub(r'(^Item)(.*)', r'\g<1> \g<2>', norm_item, flags=re.I)
        return norm_item

    # 计算TF-IDF值，返回值（特征数组，TF-IDF权重矩阵，特征权重求和+开方）
    def calc_tfidf_matrix(self, corpus):
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=0., max_df=1., use_idf=True, norm=None)
        # 将词频矩阵X统计成TF-IDF值
        tfidf = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names()
        A = tfidf.toarray()
        # 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
        TS = []
        for j in range(0, len(self.rc) - 1):
            TS.append(np.sqrt(np.sum(A[j])))
        return feature_names, A, TS


if __name__ == '__main__':
    split_file = SplitFile(r'..\2013', r'..\min-out', r'..\out\corpus_a', r'..\min-log\history.log', r'..\base_gvkey_cik_2006-2020.txt')
    start = time.time()
    split_file.walk()
    end = time.time()
    print('the consumption of total length', end - start)
