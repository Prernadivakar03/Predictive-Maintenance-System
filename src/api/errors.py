class ServiceError(Exception):
    """Framework-independent error raised by the dashboard service layer.

    The FastAPI routes translate it into an ``HTTPException`` with the same status code and detail message,
    so the service functions stay importable (and testable) without FastAPI.
    """

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
