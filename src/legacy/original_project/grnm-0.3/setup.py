from setuptools import setup, find_packages
setup(
    name = 'grnm',
    version='0.6',
    author='heay',  # 作者名字
    author_email='hyjd21@gmail.com',
    description='A comprehensive tool for gene regulatory network modeling.',  # 简短描述
    long_description=open('README.md').read(),

    packages=find_packages(),
    include_package_data=True,  # 确保读取 MANIFEST.in 文件
    package_data={
        '': ['notebooks/*.ipynb'],  # 包含 notebooks 文件夹中的所有 .ipynb 文件
    },
    python_requires='>=3.7',
    py_modules=['grnm'],

    install_requires=[
        'numpy>=1.22.0',
        'pandas==2.0.3',
        'python-louvain==0.16',
        'matplotlib==3.7.2',
        'networkx==3.2.1',
        'scikit-learn',
        'torch==2.2.1',
        'dgl',
        'umap-learn==0.5.3',
        'seaborn==0.13.2',
    ],
    keywords = 'gene regulatory networks, bioinformatics, graph models, network analysis',
    license = 'MIT',
    zip_safe = False,
)