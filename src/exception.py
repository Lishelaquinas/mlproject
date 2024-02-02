# you can give this in reqts file but usually this will be installed
import sys

'''
error  to be printed.
error_detail will be present inside sys.
exc_info will  return three parameters for error_detail the important one is exc_info gives detail about the exception.
'''
def error_message_details(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    fileName = exc_tb.tb_frame.f_code.co_filename
    message = "Error occrured in python script name [{0}] line number [{1}] error message [{2}]".format(
        fileName, exc_tb.tb_lineno,str(error))
    return message
    
class CustomException(Exception):
    def __init__ (self,error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail=error_detail)
    
    def __str__ (self):
        return self.error_message
    



