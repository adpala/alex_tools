from setuptools import setup, find_packages
import os

# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='alex_tools',
      version='0.1',
      description='tools for Alex project',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/adpala/alex_tools.git',
      author='Adrian Palacios Munoz',
      author_email='adpala93@gmail.com',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=['scikit-learn', 'numpy', 'matplotlib', 'scipy'],
      include_package_data=True,
      zip_safe=False
      )