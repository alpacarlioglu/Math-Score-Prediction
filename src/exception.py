import sys

def error_message_detail(error, error_detail: sys) -> str:
    '''
    This is a custom exception handler that captures detailed error information.
    
    Using the sys module, we can extract the file name, line number, and error message.
    This provides more context on where and why the error occurred.
    '''
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_name
    line_number = exc_tb.tb_lineno
    error_message = str(error)

    error_message = 'Error occurred in script: {} at line number: {}. Error message: {}'.format(file_name, line_number, error_message)
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        
    def __str__(self):
        return self.error_message