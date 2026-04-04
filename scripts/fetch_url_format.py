import sys

import trafilatura
import urllib.request
import urllib.error
import gzip
from trafilatura.downloads import DEFAULT_HEADERS, fetch_response, fetch_url

# 目标网页（随便一篇新闻/文章都行）
url = "https://www.sixiangjia.de/finance/Kuomintang-finance"

# 1. 下载页面
html = fetch_url(url)

# 2. 提取干净正文（自动去掉广告、导航、菜单）
content = trafilatura.extract(html) or ""

# 3. 提取标题、作者、日期等元信息
meta = trafilatura.extract_metadata(html)

# 输出看看
if meta is None:
    title = author = date = "未知"
else:
    title = getattr(meta, "title", None) or "未知"
    author = getattr(meta, "author", None) or "未知"
    date = getattr(meta, "date", None) or "未知"

print("标题:", title)
print("作者:", author)
print("日期:", date)
print("-" * 50)
print("正文:\n", content[:1000])  # 只打印前1000字
