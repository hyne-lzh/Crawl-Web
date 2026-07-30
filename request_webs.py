"""
request_webs.py — 爬取数据模块（主程序入口）
使用 playwright 爬取搜索引擎结果
用户输入问题 -> 爬取 -> 交给 extract_data.py
"""

import json
from pathlib import Path


class WebCrawler:
    """搜索引擎爬虫"""

    def __init__(self, config_path: str = "webs.json"):
        """
        初始化爬虫
        :param config_path: 搜索引擎配置文件路径
        """
        self.config_path = Path(config_path)
        self.engines: list[dict] = []
        self.raw_results: list[dict] = []  # 爬取的原始数据

    def load_engines(self) -> None:
        """从 webs.json 加载搜索引擎列表"""
        pass

    def filter_by_speed(self, min_speed: int = 0) -> list[dict]:
        """
        按 speed 过滤搜索引擎
        :param min_speed: 最低速度阈值（0-10）
        :return: 过滤后的引擎列表
        """
        pass

    async def crawl_single(self, engine: dict, query: str) -> dict:
        """
        使用 playwright 爬取单个搜索引擎
        :param engine: 搜索引擎信息 {name, url, speed}
        :param query: 用户搜索关键词
        :return: 原始爬取结果
        """
        pass

    async def crawl_all(self, query: str, min_speed: int = 5) -> list[dict]:
        """
        并发爬取所有符合条件的搜索引擎
        :param query: 用户搜索关键词
        :param min_speed: 最低速度阈值
        :return: 所有原始爬取结果列表
        """
        pass

    def run(self) -> None:
        """
        主流程入口
        1. 加载引擎列表
        2. 接收用户输入
        3. 爬取数据
        4. 交给 extract_data 处理
        """
        pass


if __name__ == "__main__":
    crawler = WebCrawler()
    crawler.run()
