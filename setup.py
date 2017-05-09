from distutils.core import setup

setup(
    name='pysdss',
    version='0.0.1',
    packages=['pysdss', 'pysdss.geostat', 'pysdss.utility', 'pysdss.database', 'pysdss.gridding', 'pysdss.filtering',
              'pysdss.multicriteria'],
    url='',
    license='',
    author='Claudio Piccinini',
    author_email='c.piccinini2@newcastle.ac.uk',
    description='Server library for database connection and analytics operations including data filtering, gridding, geostatistics, multicriteria analysis'
)
