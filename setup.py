from distutils.core import setup

setup(
    name="pysdss",
    version="0.0.1",
    packages=["pysdss","pysdss.analytics","pysdss.clustering",  "pysdss.cvision",  "pysdss.database",  "pysdss.filtering",
              "pysdss.filtering", "pysdss.geoprocessing", "pysdss.geostat", "pysdss.gridding", "pysdss.interactive",
              "pysdss.gridding",  "pysdss.interactive", "pysdss.mapserver", "pysdss.multicriteria", "pysdss.utility"],
    url="",
    license="",
    author="Claudio Piccinini",
    author_email="c.piccinini2@newcastle.ac.uk",
    description="library for database connection and analytics operations including data filtering, gridding, geostatistics, multicriteria analysis"
)
