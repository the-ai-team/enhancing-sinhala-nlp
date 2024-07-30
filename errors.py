class GeneralError(Exception):
    """Base class for other exceptions"""

    def __init__(self, message="An error occurred"):
        self.message = message
        super().__init__(self.message)


class DatasetParquetNameError(GeneralError):
    """Raised when the dataset name does not match the expected pattern"""

    def __init__(self, message="Dataset name does not match the expected pattern"):
        super().__init__(message)


class MaxChunkSizeExceededError(GeneralError):
    """Raised when the maximum chunk size is exceeded"""

    def __init__(self, message="Maximum chunk size exceeded"):
        super().__init__(message)


class DelimiterAlreadyExistsError(GeneralError):
    """Raised when the delimiter already exists"""

    def __init__(self, message="Delimiter already exists"):
        super().__init__(message)


class TranslateIOMismatchError(GeneralError):
    """Raised when the input and output count of the translation does not match"""

    def __init__(self, message="Input and output count of the translation does not match"):
        super().__init__(message)


class InvalidOutputError(GeneralError):
    """Raised when the output of a function is invalid"""

    def __init__(self, message="Invalid output"):
        super().__init__(message)


class MissingTranslationError(GeneralError, ):
    """Raised when the translation is missing"""

    def __init__(self, message="Translation is missing", i: int = None):
        super().__init__(f"{message} at index {i}" if i is not None else message)


class ReachedMaxRetriesError(GeneralError):
    """Raised when the maximum number of retries is reached"""

    def __init__(self, message="Maximum number of retries reached"):
        super().__init__(message)
