pysdss
Server library for database connection and analytics operations including data filtering, gridding,
geostatistics, and multicriteria analysis


pysdss
|
| utils.py           ### some general utility functions
|
|_______ analytics ## analytics operations
|           |
|           |__ __init__.py
|           |
|           |______ colorgrade ## colorgrade analytics operations
|                       |
|                       |__ __init__.py
|                       |
|                       |__ colorgrade.py
|                       |
|                       |__ research # this package is for research only, no production code
|                           |
|                           |__ colorgrade_comparisons.py
|
|_______ clustering ### clustering operations
|            |
|            |__ __init__.py
|
|_______ cvision     # computer vision operations
|            |
|            |_ __init__.py
|            |
|            |__ feature_extract.py
|
|
|_______ database    ### database connection and database operations
|            |
|            |__ __init__.py
|            |
|            |__ database.py
|
|_______ filtering    ### filtering operations
|            |
|            |__ __init__.py
|            |
|            |__ filter.py
|
|_______ geostat      ### geostatistics operations
|            |
|            |__ __init__.py
|
|_______ gridding      ### gridding operations
|            |
|            |__ __init__.py
|
|_______ interactive    ### interactive console
|            |
|            |__ __init__.py
|
|_______ mapserver     ## setup mapfiles for wms services
|            |
|            |__ __init.py__
|            |
|            |__ mapfile.py
|_______ multicriteria   ### multicriteria analysisi operations
|            |
|            |__ __init__.py
|            |
|            constraint
|            |
|            |__ __init.py__
|            |__ compensatory.py
|            |__ fuzzy.py
|            |
|            scale
|            |__ __init__.py
|            |__ function.py
|            |__ fuzzy.py
|            |__ linear.py
|            |__ probability.py
|            |
|            weight
|            |__ __init.py__
|            |__ pairwise
|            |__ ranking.py
|            |__ rating.py
|
|_______ utility         ### collection of spatial utility functions and modules
             |
             |__ __init__.py


