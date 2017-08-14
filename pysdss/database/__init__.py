import psycopg2
def pgconnect(**kwargs):
    """ Connect to a postgresql database
    call as pgconnect(**kwargs) where kwargs  is something like
    {"dbname": "grapes", "user": "claudio", "password": "xxx", "port":"5432", "host": "127.0.0.1"}
    :param **kwargs :   a dictionary with connection
    :return: postgresql connection
    """

    # http: // stackoverflow.com / questions / 17443379 / psql - fatal - peer - authentication - failed -for -user - dev
    #connstring = "dbname="+dbname +" user="+user +" password="+password+ " host="+host
    connstring = ""
    for i in kwargs:connstring += str(i) +"="+ kwargs[i]+ " "
    #print(connstring)
    conn = psycopg2.connect(connstring)
    return conn