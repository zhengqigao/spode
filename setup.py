from setuptools import setup, find_packages

setup(
        name='spode',
        version='0.0',
        description='A programmable photonics differentiable simulator',
        author='ZhengqiGao',
        author_email='zhenqi@mit.edu',
        url='https://github.com/zhengqigao/podis/tree/main',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        install_requires = ['numpy'],
        license = 'GPL-3.0',
        python_requires='>=3',
        packages=find_packages()
)

