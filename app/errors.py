from fastapi import HTTPException

class NoGazettesFoundError(HTTPException):
    def __init__(self):
        super().__init__(status_code=404, detail="No public gazettes were found for the given criteria.")

class CityNotFoundError(HTTPException):
    def __init__(self):
        super().__init__(status_code=404, detail="City not found in the database.")
