import pandas as pd
class MarkGenePre:
    def __init__(self, filepath):
        """
        初始化 MarkGenePre 实例并加载 CSV 数据。

        :param filepath: CSV 文件的路径
        """
        self.filepath = filepath
        self.data = None
        self.load_csv()

    def load_csv(self):
        """
        从 CSV 文件加载数据。
        """
        try:
            self.data = pd.read_csv(self.filepath,index_col = 0)
            print("CSV file loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the CSV file: {e}")
