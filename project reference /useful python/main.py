#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2022/5/30 20:17
# @Author : doFighter
import re


def getIeeeJournalFormat(bibInfo):
    """
    生成期刊文献的IEEE引用格式：{作者}, "{文章标题}," {期刊名称}, vol. {卷数}, no. {编号}, pp. {页码}, {年份}.
    :return: {author}, "{title}," {journal}, vol. {volume}, no. {number}, pp. {pages}, {year}.
    """
    # 避免字典出现null值
    if "volume" not in bibInfo:
        bibInfo["volume"] = "null"
    if "number" not in bibInfo:
        bibInfo["number"] = "null"
    if "pages" not in bibInfo:
        bibInfo["pages"] = "null"

    journalFormat = bibInfo["author"] + \
                    ", \"" + bibInfo["title"] + \
                    ",\" " + bibInfo["journal"] + \
                    ", vol. " + bibInfo["volume"] + \
                    ", no. " + bibInfo["number"] + \
                    ", pp. " + bibInfo["pages"] + \
                    ", " + bibInfo["year"] + "."

    # 对格式进行调整，去掉没有的信息，调整页码格式
    journalFormatNormal = journalFormat.replace(", vol. null", "")
    journalFormatNormal = journalFormatNormal.replace(", no. null", "")
    journalFormatNormal = journalFormatNormal.replace(", pp. null", "")
    journalFormatNormal = journalFormatNormal.replace("--", "-")
    return journalFormatNormal


def getIeeeConferenceFormat(bibInfo):
    """
    生成会议文献的IEEE引用格式：{作者}, "{文章标题}, " in {会议名称}, {年份}, pp. {页码}.
    :return: {author}, "{title}, " in {booktitle}, {year}, pp. {pages}.
    """
    conferenceFormat = bibInfo["author"] + \
                       ",\"" + bibInfo["title"] + ",\" " + \
                       "in " + bibInfo["booktitle"] + \
                       ", " + bibInfo["year"] + \
                       ", pp. " + bibInfo["pages"] + "."

    # 对格式进行调整，，调整页码格式
    conferenceFormatNormal = conferenceFormat.replace("--", "-")
    return conferenceFormatNormal


def getIeeeFormat(bibInfo):
    """
    本函数用于根据文献类型调用相应函数来输出ieee文献引用格式
    :param bibInfo: 提取出的BibTeX引用信息
    :return: ieee引用格式
    """
    if "journal" in bibInfo:  # 期刊论文
        return getIeeeJournalFormat(bibInfo)
    elif "booktitle" in bibInfo:  # 会议论文
        return getIeeeConferenceFormat(bibInfo)


# 查找名，并按格式进行缩写
def capitalLetter(name):
    resName = ''
    for i in name:
        if i.isupper():
            resName += i + '. '

    return resName


# 按照bib格式，调整作者的姓名缩写形式
def nameModefy(name):
    nameList = name.split(' and ')
    resNames = []
    for index in range(len(nameList)):
        if index > 1:
            break
        names = nameList[index].split(',')
        if len(names) < 2:
            continue

        for i in range(len(names)):
            names[i] = names[i].strip()

        resName = capitalLetter(names[1]) + names[0]
        resNames.append(resName)

    result = ' and '.join(resNames)

    if len(nameList) > 2:
        result += ', et al.'

    return result


def inforDir(bibtex):
    # pattern = "[\w]+={[^{}]+}"   用正则表达式匹配符合 ...={...} 的字符串
    pattern1 = "[\w]+="  # 用正则表达式匹配符合 ...= 的字符串
    pattern2 = "{[^{}]+}"  # 用正则表达式匹配符合 内层{...} 的字符串

    # 找到所有的...=，并去除=号
    result1 = re.findall(pattern1, bibtex)
    for index in range(len(result1)):
        result1[index] = re.sub('=', '', result1[index])
    # 找到所有的{...}，并去除{和}号
    result2 = re.findall(pattern2, bibtex)
    for index in range(len(result2)):
        result2[index] = re.sub('\{', '', result2[index])
        result2[index] = re.sub('\}', '', result2[index])

    # 创建BibTeX引用字典，归档所有有效信息
    infordir = {}
    for index in range(len(result1)):
        if result1[index] == 'author':
            infordir[result1[index]] = nameModefy(result2[index])
        else:
            infordir[result1[index]] = result2[index]
    return infordir


def inputBibTex():
    """
    在这里输入BibTeX格式的文献引用信息
    :return:提取出的BibTeX引用信息
    """
    bibtex = []

    print("请输入BibTeX格式的文献引用：")
    i = 0
    while i < 15:  # 观察可知BibTeX格式的文献引用不会多于15行
        lines = input()
        if len(lines) == 0:  # 如果输入空行，则说明引用内容已经输入完毕
            break
        else:
            bibtex.append(lines)
        i += 1
    return inforDir("".join(bibtex))


if __name__ == '__main__':
    bibInfo = inputBibTex()  # 获得BibTeX格式的文献引用
    print(getIeeeFormat(bibInfo))  # 输出ieee格式

