"""
wash_data.py — 数据清洗与过滤模块
使用 sklearn 进行广告检测、数据清洗
接收 extract_data.py 的提取结果 -> 过滤 -> 展示 & 保存
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DataWasher:
    """数据清洗器"""

    # 广告语料（正样本）
    AD_CORPUS: list[str] = [
        "广告 推广 特价 限时优惠 点击购买 立即抢购",
        "sponsored ad promotion buy now discount deal offer",
        "免费领取 限时抢购 超低价 大促 满减 包邮 秒杀",
        "推广链接 商业推广 为您推荐 热门商品 爆款推荐",
        "advertisement paid result 推广 广告位 竞价排名",
        "优惠券 折扣 促销 厂家直销 批发价 一折起 清仓",
        "立即咨询 在线客服 免费试用 注册即送 名额有限",
        "sponsored content 品牌推广 合作伙伴 推荐商家",
    ]

    # 正常搜索结果语料（负样本）
    NORMAL_CORPUS: list[str] = [
        "维基百科 百度百科 官方文档 技术博客 学术论文",
        "wikipedia documentation tutorial guide reference manual",
        "新闻 资讯 报道 文章 百科 问答 论坛 社区 讨论",
        "官方 官网 政府 教育 研究 学术 期刊 论文 图书馆",
        "开源项目 GitHub 代码 编程 技术分享 开发者文档",
        "百科 知识 历史 科学 文化 地理 医学 法律 经济",
        "API reference specification standard protocol RFC",
        "教程 入门 指南 手册 文档 帮助 常见问题 FAQ",
    ]

    # 广告关键词（规则辅助）
    AD_KEYWORDS: list[str] = [
        "广告", "推广", "促销", "限时", "抢购", "秒杀", "特价",
        "优惠", "折扣", "满减", "包邮", "免费领取", "立即购买",
        "sponsored", "ad", "promotion",
    ]

    def __init__(self, output_dir: str = "results"):
        """
        初始化清洗器
        :param output_dir: 结果保存目录
        """
        self.extracted_data: list[dict] = []
        self.cleaned_data: list[dict] = []
        self.ad_model: LogisticRegression | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------- 接收数据 ---------------------

    def receive(self, extracted_data: list[dict]) -> None:
        """
        接收 extract_data.py 传来的提取数据
        :param extracted_data: 提取后的结构化数据
        """
        self.extracted_data = extracted_data
        self.cleaned_data = []

    # --------------------- 广告检测 ---------------------

    @staticmethod
    def _join_item(item: dict) -> str:
        """将单条结果的 title/snippet 合并为一段文本"""
        parts = []
        if item.get("title"):
            parts.append(item["title"])
        if item.get("snippet"):
            parts.append(item["snippet"])
        return " ".join(parts)

    @staticmethod
    def _rule_based_ad_check(text: str) -> bool:
        """基于规则的广告检测（快速预筛）"""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in DataWasher.AD_KEYWORDS)

    def train_ad_model(self) -> None:
        """
        使用 sklearn 训练广告检测模型
        TF-IDF 向量化 + LogisticRegression 分类器
        """
        # 构造训练数据
        X_texts = self.AD_CORPUS + self.NORMAL_CORPUS
        y = [1] * len(self.AD_CORPUS) + [0] * len(self.NORMAL_CORPUS)

        # TF-IDF 向量化
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            sublinear_tf=True,
        )
        X = self.vectorizer.fit_transform(X_texts)

        # 训练逻辑回归分类器
        self.ad_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced",
        )
        self.ad_model.fit(X, y)

    def detect_ads(self, item: dict) -> bool:
        """
        检测单条结果是否为广告
        先规则预判，再用 sklearn 模型确认
        :param item: 单条搜索结果 {title, url, snippet}
        :return: True 表示是广告
        """
        text = self._join_item(item)
        if not text.strip():
            return False

        ad_score = 0

        # 1. 规则检测
        if self._rule_based_ad_check(text):
            ad_score += 3

        # 2. URL 特征检测（广告链接常有 query 参数特征）
        url = item.get("url", "")
        if url and any(kw in url.lower() for kw in ["ad", "sponsor", "promote", "promotion", "track"]):
            ad_score += 2

        # 3. sklearn 模型预测
        if self.ad_model is not None and self.vectorizer is not None:
            try:
                vec = self.vectorizer.transform([text])
                proba = self.ad_model.predict_proba(vec)[0, 1]
                if proba > 0.6:
                    ad_score += 4
            except Exception:
                pass  # 向量化失败则跳过模型判断

        # 综合判定：得分 >= 4 视为广告
        return ad_score >= 4

    def filter_ads(self) -> None:
        """过滤所有广告结果"""
        if self.ad_model is None:
            self.train_ad_model()

        for engine_data in self.extracted_data:
            results = engine_data.get("results", [])
            engine_data["results"] = [
                item for item in results if not self.detect_ads(item)
            ]
        self.cleaned_data = self.extracted_data

    # --------------------- 去重 ---------------------

    def deduplicate(self, threshold: float = 0.85) -> None:
        """
        去重：删除文本相似度 >= threshold 的重复结果
        使用 TF-IDF + cosine_similarity
        :param threshold: 相似度阈值（0-1），超过视为重复
        """
        if not self.cleaned_data:
            return

        # 收集所有结果（跨引擎去重）
        all_items: list[dict] = []
        for engine_data in self.cleaned_data:
            all_items.extend(engine_data.get("results", []))

        if len(all_items) <= 1:
            return

        # 提取文本并向量化
        texts = [self._join_item(item) for item in all_items]
        vec = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(texts)
        sim_matrix = cosine_similarity(vec)

        # 标记要去掉的索引
        n = len(all_items)
        to_remove: set[int] = set()
        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                if sim_matrix[i, j] >= threshold:
                    # 保留文本较长的那条（信息更丰富）
                    if len(texts[i]) >= len(texts[j]):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break  # i 被移除，跳到下一个 i

        # 全局去重：先找出所有唯一项，再按引擎重新分组
        # 建立 全局索引 -> (引擎索引, 引擎内索引) 的映射
        global_idx = 0
        index_map: list[tuple[int, int]] = []  # [(engine_i, item_j), ...]
        for ei, engine_data in enumerate(self.cleaned_data):
            for ej, item in enumerate(engine_data.get("results", [])):
                index_map.append((ei, ej))

        # 为每个引擎收集保留的结果
        new_results: dict[int, list[dict]] = {ei: [] for ei in range(len(self.cleaned_data))}
        for gi, (ei, ej) in enumerate(index_map):
            if gi not in to_remove:
                new_results[ei].append(self.cleaned_data[ei]["results"][ej])

        for ei, engine_data in enumerate(self.cleaned_data):
            engine_data["results"] = new_results[ei]

    # --------------------- 相关性排序 ---------------------

    def rank_by_relevance(self, query: str) -> None:
        """
        按与查询的相关性重新排序
        使用 TF-IDF + cosine_similarity 计算每条结果与 query 的相似度
        :param query: 原始搜索关键词
        """
        if not self.cleaned_data:
            return

        for engine_data in self.cleaned_data:
            results = engine_data.get("results", [])
            if len(results) <= 1:
                continue

            texts = [self._join_item(item) for item in results]
            # 将 query 放在第一位参与向量化
            all_texts = [query] + texts
            vec = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(all_texts)
            # 计算 query 与每条结果的余弦相似度
            sim_scores = cosine_similarity(vec[0:1], vec[1:]).flatten()

            # 按相似度降序排列
            scored = list(zip(results, sim_scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            engine_data["results"] = [item for item, _ in scored]

    # --------------------- 主清洗流程 ---------------------

    def wash(self, query: str) -> list[dict]:
        """
        完整清洗流程：去广告 -> 去重 -> 排序
        :param query: 原始搜索关键词
        :return: 清洗后的结果
        """
        print("[wash] 开始清洗数据...")

        # 1. 过滤广告
        print("  [1/3] 广告检测与过滤...")
        self.filter_ads()
        total_before = sum(len(d.get("results", [])) for d in self.cleaned_data)

        # 2. 去重
        print("  [2/3] 相似结果去重...")
        self.deduplicate()
        total_after = sum(len(d.get("results", [])) for d in self.cleaned_data)

        # 3. 相关性排序
        print("  [3/3] 相关性排序...")
        self.rank_by_relevance(query)

        print(f"  [完成] 清洗前 {len(self.extracted_data)} 个引擎，"
              f"清洗后保留 {total_after} 条结果（去重前 {total_before} 条）")
        return self.cleaned_data

    # --------------------- 展示 ---------------------

    def display(self) -> None:
        """在终端展示清洗后的结果给用户"""
        if not self.cleaned_data:
            print("无结果可展示")
            return

        print("\n" + "=" * 70)
        print("                    搜索结果")
        print("=" * 70)

        total = 0
        for engine_data in self.cleaned_data:
            engine_name = engine_data.get("engine", "未知引擎")
            results = engine_data.get("results", [])
            if not results:
                continue
            print(f"\n  [{engine_name}] ({len(results)} 条)")
            print("  " + "-" * 60)
            for i, item in enumerate(results, 1):
                title = item.get("title", "无标题")
                url = item.get("url", "")
                snippet = item.get("snippet", "")
                print(f"  {i:2d}. {title}")
                if url:
                    print(f"       {url}")
                if snippet:
                    # 截断过长的摘要
                    display_snippet = snippet[:120] + "..." if len(snippet) > 120 else snippet
                    print(f"       {display_snippet}")
                print()
                total += 1

        print("-" * 70)
        print(f"  共 {total} 条结果\n")

    # --------------------- 保存 ---------------------

    def save_json(self, filename: str | None = None) -> str:
        """
        保存结果到本地 JSON 文件
        :param filename: 文件名，默认使用时间戳
        :return: 保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"

        filepath = self.output_dir / filename
        output_data = {
            "export_time": datetime.now().isoformat(),
            "total_engines": len(self.cleaned_data),
            "total_results": sum(len(d.get("results", [])) for d in self.cleaned_data),
            "data": self.cleaned_data,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"[保存] JSON 已保存至: {filepath}")
        return str(filepath)

    def save_csv(self, filename: str | None = None) -> str:
        """
        保存结果到本地 CSV 文件
        :param filename: 文件名，默认使用时间戳
        :return: 保存的文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.csv"

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["引擎", "序号", "标题", "链接", "摘要"])
            for engine_data in self.cleaned_data:
                engine_name = engine_data.get("engine", "未知")
                for i, item in enumerate(engine_data.get("results", []), 1):
                    writer.writerow([
                        engine_name,
                        i,
                        item.get("title", ""),
                        item.get("url", ""),
                        item.get("snippet", ""),
                    ])

        print(f"[保存] CSV 已保存至: {filepath}")
        return str(filepath)

    # --------------------- 主入口 ---------------------

    def run(self, extracted_data: list[dict], query: str) -> None:
        """
        清洗模块主流程
        1. 接收数据
        2. 清洗过滤
        3. 展示结果
        4. 保存到本地
        """
        self.receive(extracted_data)
        self.wash(query)
        self.display()
        self.save_json()
        self.save_csv()


# --------------------- 独立测试 ---------------------

if __name__ == "__main__":
    # 模拟 extract_data.py 传来的数据
    mock_data = [
        {
            "engine": "百度",
            "query": "Python 教程",
            "results": [
                {
                    "title": "Python 官方教程",
                    "url": "https://docs.python.org/zh-cn/3/tutorial/",
                    "snippet": "Python 是一门简单易学、功能强大的编程语言。本教程适合初学者。",
                },
                {
                    "title": "Python 入门 - 菜鸟教程",
                    "url": "https://www.runoob.com/python/",
                    "snippet": "Python 基础教程，包含 Python 语法、数据类型等知识。",
                },
                {
                    "title": "【限时优惠】Python 编程课 - 特价秒杀",
                    "url": "https://example-ad.com/python?promotion=1",
                    "snippet": "广告推广：限时优惠！Python 从入门到精通，点击立即购买，名额有限！",
                },
                {
                    "title": "Python 入门 - 廖雪峰的官方网站",
                    "url": "https://www.liaoxuefeng.com/wiki/python",
                    "snippet": "廖雪峰 Python 教程，涵盖基础、函数、模块、面向对象等内容。",
                },
                {
                    "title": "Python 教程 - W3Schools",
                    "url": "https://www.w3schools.com/python/",
                    "snippet": "Well organized and easy to understand Web building tutorials.",
                },
            ],
        },
        {
            "engine": "必应搜索",
            "query": "Python 教程",
            "results": [
                {
                    "title": "Python 基础教程 | 菜鸟教程",
                    "url": "https://www.runoob.com/python/",
                    "snippet": "Python 基础教程，包含 Python 语法、数据类型、条件语句等知识。",
                },
                {
                    "title": "【包邮】Python 编程书籍 - 爆款推荐",
                    "url": "https://shop-ad.com/python-book?sponsored=1",
                    "snippet": "推广链接：Python 学习书籍全套，厂家直销，满减包邮，立即抢购！",
                },
                {
                    "title": "Python Tutorial - Real Python",
                    "url": "https://realpython.com/",
                    "snippet": "Learn Python online: Python tutorials for developers of all skill levels.",
                },
            ],
        },
    ]

    washer = DataWasher()
    washer.run(mock_data, "Python 教程")
