"""
extract_data.py — 数据提取模块
使用 bs4 (BeautifulSoup4) 从原始 HTML 中提取内容
接收 request_webs.py 的爬取结果 -> 提取 -> 交给 wash_data.py
"""


class DataExtractor:
    """HTML 数据提取器"""

    def __init__(self):
        """初始化提取器"""
        self.raw_data: list[dict] = []
        self.extracted_data: list[dict] = []

    def receive(self, raw_data: list[dict]) -> None:
        """
        接收 request_webs.py 传来的原始爬取数据
        :param raw_data: 原始爬取结果列表
        """
        pass

    def extract_text(self, html: str) -> str:
        """
        从 HTML 中提取纯文本内容
        :param html: HTML 字符串
        :return: 纯文本内容
        """
        pass

    def extract_links(self, html: str) -> list[str]:
        """
        从 HTML 中提取所有链接
        :param html: HTML 字符串
        :return: 链接列表
        """
        pass

    def extract_titles(self, html: str) -> list[str]:
        """
        从 HTML 中提取所有标题
        :param html: HTML 字符串
        :return: 标题列表
        """
        pass

    def extract_snippets(self, html: str) -> list[str]:
        """
        从 HTML 中提取搜索结果摘要片段
        :param html: HTML 字符串
        :return: 摘要片段列表
        """
        pass

    def process_all(self) -> list[dict]:
        """
        对所有原始数据进行结构化提取
        :return: 提取后的结构化数据列表
        格式: [
            {
                "engine": "百度",
                "query": "搜索关键词",
                "results": [
                    {"title": "...", "url": "...", "snippet": "..."},
                    ...
                ]
            },
            ...
        ]
        """
        pass

    def export(self) -> list[dict]:
        """
        导出提取结果，交给 wash_data.py
        :return: 提取后的结构化数据
        """
        pass


if __name__ == "__main__":
    extractor = DataExtractor()
    # 由 request_webs.py 调用，此处为独立测试入口
