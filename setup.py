import re
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('m5meta.py', 'r') as fv:
    for l in fv:
        m = re.search("__version__[\s]*=[\s]*'(?P<version>[0-9]+(\.[0-9]+)+)'", l)
        if m is not None:
            version = m.group('version')
            break


setuptools.setup(
    name = 'm5meta',
    version = version,
    author = 'Eric Smith',
    author_email = 'spacewar@gmail.com',
    description = 'M5 Meta Assembler',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/brouhaha/m5meta',
    py_modules = ['m5meta'],
    install_requires = ['pyparser', 'm5pre'],
    classifiers = [
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Assemblers',
    ],
    python_requires='>=3.7.0',
)
