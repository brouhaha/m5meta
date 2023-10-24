import re
import setuptools

import m5meta

with open('README.md', 'r') as fh:
    long_description = fh.read()
    long_description_content_type = 'text/markdown'

def python_classifiers(min_python_version):
    base = 'Programming Language :: Python'
    c = [base]
    pvt = tuple([int(n) for n in min_python_version.split('.')])
    for i in range(1, len(pvt)+1):
        c.append(base + " :: " + '.'.join([str(p) for p in pvt[:i]]))
    return c

classifiers = (python_classifiers(m5meta.__min_python_version__) +
               [
                   'Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                   'Operating System :: OS Independent',
                   'Topic :: Software Development :: Assemblers',
               ])

setuptools.setup(
    name = 'M5Meta',
    version = m5meta.__version__,
    author = m5meta.__author__,
    author_email = m5meta.__email__,
    description = m5meta.__description__,
    long_description = long_description,
    long_description_content_type = long_description_content_type,
    url = m5meta.__url__,
    py_modules = ['m5meta'],
    install_requires = ['pyparser', 'm5pre'],
    python_requires='>=' + m5meta.__min_python_version__,
    classifiers = classifiers,
)
