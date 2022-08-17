from setuptools import setup, find_packages

setup(
        name='spode',
        version='0.0.2',
        description='A simulator with programmable photonics and differentiability emphasis',
        author='ZhengqiGao',
        author_email='zhenqi@mit.edu',
        url='https://github.com/zhengqigao/spode',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        install_requires = ['numpy'],
        license = 'GPL-3.0',
        package_data ={'spode': ['core/model.json']},
        python_requires='>=3',
        packages=find_packages()
)

